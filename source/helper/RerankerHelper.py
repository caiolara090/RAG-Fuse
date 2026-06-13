from pathlib import Path
from omegaconf import OmegaConf
from source.helper.Helper import Helper
import pickle
import logging
from transformers import AutoModel
from tqdm import tqdm
import torch
from .RankingAggregationHelper import compute_inv_propesity, psprecision, psndcg
from sklearn.preprocessing import MultiLabelBinarizer
from ranx import evaluate, Qrels, Run
from scipy.sparse import csr_matrix
import pandas as pd


class RerankerHelper(Helper):
    def __init__(self, params):
        super(RerankerHelper, self).__init__()
        self.params = params
        self.samples = self._load_samples()
        self.relevance_map = self._load_relevance_map()
        self.label_cls = self._load_labels_cls()
        self.text_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()
        logging.basicConfig(level=logging.INFO)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )

        # 2. Load the model
        self.model = AutoModel.from_pretrained(
            'jinaai/jina-reranker-v3',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        self.model.to(self.device)
        self.model.eval()

    def run(self):
        if self.params.data.label_enhancement != "LLM":
            logging.error('Cannot rerank without label descriptions')
            exit(1)

        rankings = []
        top_k = self.params.rerank.top_k

        sample_map = {x['text_idx']: x for x in self.samples}
        for fold_idx in self.params.data.folds:
            logging.info(
                f"Reranking {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling "
                f"self.params\n {OmegaConf.to_yaml(self.params)}\n")
            
            old_ranking = self._load_ranking(fold_idx) #ranking is a dict of keys [text_ID] -> a dict of label relevancy with keys 'label_ID'
            descriptions = self._load_label_descriptions(fold_idx) #flat array

            labels_ids = []
            for sample in self.samples:
                labels_ids.append(sample['labels_ids'])

            #Do I need this? This is for extreme label classification only?
            mlb = MultiLabelBinarizer(sparse_output=True)
            labels = mlb.fit_transform(labels_ids)
            inv_propensity = compute_inv_propesity(labels, self.params.data.propensity.A,
                                                   self.params.data.propensity.B)
            
            reranked = {}

            for text, rank in tqdm(old_ranking.items(), total=len(old_ranking)):
                text_idx = int(text.split('_')[1])
                flat = sorted(
                    [
                        { 
                            'score': score,
                            'label': label,
                            'desc': descriptions[int(label.split('_')[1])]
                        } 
                        for label, score in rank.items()
                        if int(label.split('_')[1]) in descriptions
                    ],
                    key=lambda x: x['score'], 
                    reverse=True 
                )
                flat = flat[:top_k]

                frame, phrase = sample_map[text_idx]['text'].split(' : ', maxsplit=1)
                query = f"Alvo: {frame} | Contexto: {phrase}"

                results = self.model.rerank(query, [x['desc'] for x in flat])

                reranked[text] = { flat[r['index']]['label']: r['relevance_score'] for r in results }
            
            result = self._eval_combined_ranking(
                reranked,
                self.relevance_map,
                inv_propensity.shape[0],
                inv_propensity,
                self.params.eval.thresholds
            )
            result["fold_idx"] = fold_idx

            self._checkpoint_result(result, fold_idx)
            self._checkpoint_ranking(reranked, fold_idx)

            rankings.append(reranked)
    
    def _load_ranking(self, fold_idx):
        with open(
                f"{self.params.ranking.dir}Aggregated_"
                f"{self.params.model.name}_{self.params.data.name}/Aggregated_"
                f"{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk",
                "rb") as ranking_file:
            return pickle.load(ranking_file)
        
    def _load_label_descriptions(self, fold_idx):
        with open(f"resource/dataset/{self.params.data.name}/fold_{fold_idx}/labels_descriptions.pkl",
                      "rb") as labels_desc_file:
                self.labels_descriptions = pickle.load(labels_desc_file)
                logging.info(f"Loaded {len(self.labels_descriptions)} labels descriptions")
        return self.labels_descriptions
    
    def _checkpoint_ranking(self, ranking, fold_idx):
        ranking_dir = f"{self.params.ranking.dir}Reranked_{self.params.model.name}_{self.params.data.name}/"
        Path(ranking_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving ranking {fold_idx} on {ranking_dir}")
        with open(f"{ranking_dir}Reranked_{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk",
                  "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)

    def _checkpoint_result(self, result, fold_idx):
        result_dir = f"{self.params.result.dir}Reranked_{self.params.model.name}_{self.params.data.name}/"
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving result for fold {fold_idx} on {result_dir}")
        pd.DataFrame([result]).to_csv(
            f"{result_dir}Reranked_{self.params.model.name}_{self.params.data.name}_{fold_idx}.rts",
            sep='\t', index=False, header=True)

    def _eval_combined_ranking(self, ranking, relevance_map, num_labels, inv_propesities, thresholds):
        # Get propensity scored metrics
        ps_results = self._compute_ps_metrics(ranking, relevance_map, num_labels, inv_propesities, thresholds)

        # Get traditional ranking metrics
        tr_results = self._compute_tr_metrics(ranking)

        # Combine results into a single dictionary
        results = {}
        results.update(ps_results)
        results.update(tr_results)

        return results

    #TODO taken from aggregation helper, maybe extract?
    def _compute_ps_metrics(self, ranking, relevance_map, num_labels, inv_propesities, thresholds):
        text_ids_map = {k: v for v, k in enumerate(ranking.keys())}
        p_rows, p_cols, p_scores = [], [], []
        t_rows, t_cols, t_scores = [], [], []
        for text_idx, labels_scores in ranking.items():
            for label_idx, score in labels_scores.items():
                if int(label_idx.split("_")[-1]) >= 0:
                    p_rows.append(text_ids_map[text_idx])
                    p_cols.append(int(label_idx.split("_")[-1]))
                    p_scores.append(score)

            for label_idx in relevance_map[text_idx]:
                t_rows.append(text_ids_map[text_idx])
                t_cols.append(int(label_idx.split("_")[-1]))
                t_scores.append(1.0)

        pred = csr_matrix((p_scores, (p_rows, p_cols)), shape=(len(text_ids_map), num_labels))
        true = csr_matrix((t_scores, (t_rows, t_cols)), shape=(len(text_ids_map), num_labels))

        return self.__compute_ps_metrics(pred, true, inv_propesities, thresholds)

    #TODO also taken from aggegation helper
    def __compute_ps_metrics(self, pred, true, inv_propesities, thresholds):
        psprecisions = psprecision(pred, true, inv_propesities, k=thresholds[-1])
        psndcgs = psndcg(pred, true, inv_propesities, k=thresholds[-1])
        results = {}
        for k in thresholds:
            results[f"psnDCG@{k}"] = round(100 * psndcgs[k - 1], 1)

        for k in thresholds:
            results[f"psprecision@{k}"] = round(100 * psprecisions[k - 1], 1)

        return results

    #TODO also taken by aggr helper
    def _compute_tr_metrics(self, ranking):
        result = evaluate(
            Qrels(
                {key: value for key, value in self.relevance_map.items() if key in ranking.keys()}
            ),
            Run(ranking),
            self.metrics
        )
        return {k: round(100 * v, 1) for k, v in result.items()}