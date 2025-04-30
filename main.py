###
# 0 is ungramamtical/incoherent
# 1 is grammatical/coherent
###

from transformers import pipeline, logging  # type: ignore
from tqdm import tqdm
from typing import Callable, List, Dict, Literal
import pandas as pd
from inflect import engine
from dataclasses import dataclass, field

from pluralizer import Pluralizer  # type: ignore

logging.set_verbosity_error()

plural = Pluralizer.pluralize
inflect = engine()  # type: ignore

Prediction = Dict[str, float]
PredictionMulti = Dict[str, List[float]]
TestFunction = Callable[[str], str]
ScoreFunction = Callable[[Prediction], float]


def get_nouns(filename: str) -> List[str]:
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip()]


@dataclass
class ScoringConfig:
    polarity: Literal["reward", "punish"] = "reward"
    labels: List[str] = field(default_factory=lambda: ["grammatical"])
    combination_mode: Literal["mean", "sum"] = "mean"


@dataclass
class NounTest:
    test_type: str
    pipeline: Literal["grammar", "zero-shot"] = "grammar"
    label_candidates: List[str] = field(default_factory=lambda: ["grammatical", "ungrammatical"])
    scoring_config: ScoringConfig = field(default_factory=lambda: ScoringConfig())
    test_funcs: List[TestFunction] = field(default_factory=lambda: [lambda noun: str(noun)])


NOUN_TESTS: List[NounTest] = [
    NounTest(
        test_type="Eventive",
        test_funcs=[
            lambda noun: f"The {noun} took a long time",
            lambda noun: f"The {noun} occurred unexpectedly",
        ],
    ),
    NounTest(
        test_type="Pluralia Tantum",
        scoring_config=ScoringConfig("punish", ["ungrammatical"]),
        test_funcs=[
            lambda noun: f"That {noun} is hers",
            lambda noun: f"That {noun} belongs to her",
        ],
    ),
    NounTest(
        test_type="Singularia Tantum",
        scoring_config=ScoringConfig("punish", ["ungrammatical"]),
        test_funcs=[
            lambda noun: f"Those {inflect.plural_noun(noun)} are hers",
            lambda noun: f"Those {inflect.plural_noun(noun)} belong to her",
        ],
    ),
    NounTest(
        test_type="Collective",
        test_funcs=[
            lambda noun: f"This {noun} consists of many members",
            lambda noun: f"The {noun} gathered together",
        ],
    ),
    NounTest(
        test_type="Common",
        test_funcs=[
            lambda noun: f"{inflect.a(noun)} is useful",
            lambda noun: f"Any {noun} will do",
        ],
    ),
    NounTest(
        test_type="Count",
        test_funcs=[
            lambda noun: f"One {noun}, two {inflect.plural_noun(noun)}",
            lambda noun: f"Several {inflect.plural_noun(noun)} were counted",
        ],
    ),
    NounTest(
        test_type="Mass",
        test_funcs=[
            lambda noun: f"Some {noun} was left",
            lambda noun: f"How many {noun} are there",
        ],
    ),
    NounTest(  # punish if verb. punish more if both transitive and intransitive
        test_type="Verbalizable",
        test_funcs=[
            lambda noun: f"John loves to {noun} all the time",
            lambda noun: f"I want to {noun} this thing",
            lambda noun: f"I will {noun} her something tomorrow",
        ],
        scoring_config=ScoringConfig("punish", ["grammatical"], combination_mode="sum"),
    ),
    NounTest(
        test_type="Derived",
        pipeline="zero-shot",
        scoring_config=ScoringConfig("reward", ["no_suffix"]),
        label_candidates=["contains_suffix", "no_suffix"],
        test_funcs=[
            lambda noun: f"Input word: {noun}",
        ],
    ),
    NounTest(
        test_type="Usually Verb",
        pipeline="zero-shot",
        scoring_config=ScoringConfig("punish", ["usually_verb"]),
        label_candidates=["usually_verb", "usually_noun"],
        test_funcs=[
            lambda noun: f"Is the input word usually a noun or usually a verb: {noun}",
        ],
    ),
]


class Tester:
    def __init__(self, input: str | List[str], n_nouns: int = 0):
        nouns = input if isinstance(input, list) else get_nouns(input)
        end = n_nouns if n_nouns > 0 else len(nouns)
        self.nouns = nouns[:end]
        self.classifiers = {
            "grammar": pipeline(
                task="text-classification",
                model="textattack/roberta-base-CoLA",
                model_kwargs={"id2label": {0: "ungrammatical", 1: "grammatical"}},
            ),
            "zero-shot": pipeline(
                task="zero-shot-classification",
                model="roberta-large-mnli",
            ),
        }

    def whole_shebang(self):
        test_cases_df = self.generate_test_cases()
        predictions_df = self.run_test_cases(test_cases_df)
        results_df = self.score_tests(predictions_df)
        summary_df = self.calc_and_save_results(results_df)
        self.print_summary(summary_df)

    def print_summary(self, summary_df: pd.DataFrame) -> None:
        print("Most nouny nouns:")
        print(summary_df.sort_values(by="total", ascending=False).head(10))
        print("Least nouny nouns 👎👎👎:")
        print(summary_df.sort_values(by="total", ascending=True).head(10))

    def generate_test_cases(self) -> pd.DataFrame:
        print(f"Generating test cases for {len(self.nouns)} nouns")
        rows = []
        for noun in self.nouns:
            for test in NOUN_TESTS:
                for i, test_func in enumerate(test.test_funcs):
                    rows.append(
                        {
                            "noun": noun,
                            "test_type": test.test_type,
                            "test_id": f"{test.test_type}_{i}",
                            "input_sentence": test_func(noun),
                            "pipeline": test.pipeline,
                            "label_candidates": test.label_candidates,
                            "scoring_config": test.scoring_config,
                        }
                    )
        return pd.DataFrame(rows)

    def run_test_cases(self, test_cases_df: pd.DataFrame) -> pd.DataFrame:
        print(f"Running {len(test_cases_df)} test cases")

        # List to store all results
        all_results = []

        # Convert label_candidates lists to tuples for grouping
        tmp = test_cases_df.copy(deep=True)
        tmp["label_candidates"] = tmp["label_candidates"].apply(lambda x: tuple(x))

        pipe_groups = tqdm(
            tmp.groupby(["test_type", "label_candidates", "pipeline"]),
            desc="Test Groups",
            position=1,
            colour="green",
        )
        for (test_type, labels_tuple, pipeline), group in pipe_groups:

            sentences = group["input_sentence"].values.tolist()
            input_data = iter(tqdm(sentences, desc=f"{test_type:17}", position=0, smoothing=1))

            classifier = self.classifiers[pipeline]
            kwargs = {"candidate_labels": list(labels_tuple)} if pipeline == "zero-shot" else {}

            # do the classification
            preds = classifier(input_data, batch_size=128, **kwargs)

            for pred, idx in zip(preds, group.index):
                row = test_cases_df.iloc[idx].to_dict()  # Get original row data

                # Add predictions
                row["pred_labels"] = pred["labels"] if "labels" in pred else [pred["label"]]
                row["pred_scores"] = pred["scores"] if "scores" in pred else [pred["score"]]

                all_results.append(row)

        return pd.DataFrame(all_results)

    def score_tests(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        print(f"Scoring {len(predictions_df)} tests")
        results = []
        for _, row in predictions_df.iterrows():
            # get scoring config
            scoring_config = row["scoring_config"]

            # filter predictions
            predictions = zip(row["pred_labels"], row["pred_scores"])
            counted_predictions = list(
                filter(lambda ls: ls[0] in scoring_config.labels and ls[1] >= 0.5, predictions)
            )
            assert len(counted_predictions) <= 1, f"Multiple positive labels: {row['pred_labels']}"

            # calculate score
            row = row.copy()
            score = counted_predictions[0][1] if counted_predictions else 0
            polarity = 1 if scoring_config.polarity == "reward" else -1
            row["score"] = polarity * score

            results.append(row)

        df = pd.DataFrame(results)
        df.to_csv("scored_test_results.csv")
        return df

    def calc_and_save_results(self, results_df: pd.DataFrame | None = None) -> pd.DataFrame:
        print("Calculating and saving results")
        if results_df is None:
            results_df = pd.read_csv("scored_test_results.csv")

        # Get aggregation function for each test type
        agg_funcs = {test.test_type: test.scoring_config.combination_mode for test in NOUN_TESTS}

        # Group by noun and test_type, then aggregate using the appropriate function
        test_scores = []
        for test_type, func_name in agg_funcs.items():
            scores = (
                results_df[results_df["test_type"].str.startswith(test_type)]
                .groupby("noun")["score"]
                .agg(func_name)
                .rename(test_type)
            )
            test_scores.append(scores)

        # Combine all test type scores into one dataframe
        summary = pd.concat(test_scores, axis=1).fillna(0).sort_index(axis=1)

        # add total score
        summary.insert(0, "total", summary.sum(axis=1))
        summary.to_csv("results_summary.csv", float_format="%.8f")

        # Save both detailed (with individual test scores) and summary results
        detailed = pd.concat(
            [summary, results_df.pivot(index="noun", columns="test_id", values="score").fillna(0)],
            axis=1,
        )

        detailed.to_csv("results_detailed.csv", float_format="%.8f")

        sorted_results = summary.sort_values("total", ascending=False)
        sorted_results.to_csv("results_sorted.csv", float_format="%.8f")
        return summary


if __name__ == "__main__":
    tester = Tester("nounlist.txt")
    tester.whole_shebang()
