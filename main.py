from transformers import pipeline  # type: ignore
import csv
from tqdm import tqdm
from typing import Callable, List, Dict, TypedDict, Optional, Union
import pandas as pd
import inflect

from pluralizer import Pluralizer  # type: ignore

plural = Pluralizer.pluralize
inflect = inflect.engine()  # type: ignore

TestFunction = Callable[[str], str]
HeuristicFunction = Callable[[str], bool]


def get_nouns(filename: str) -> list[str]:
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip()]


TESTS: Dict[str, List[TestFunction]] = {
    # "Agentive": [
    #     lambda noun: f"The {noun} did his job",
    # ],
    "Eventive": [
        lambda noun: f"The {noun} took a long time",
        lambda noun: f"The {noun} occurred unexpectedly",
    ],
    "Pluralia Tantum": [
        lambda noun: f"{inflect.a(noun)}",
        lambda noun: f"One {noun} was enough",
    ],
    "Animate/Inanimate": [
        lambda noun: f"The {noun} lives",
        lambda noun: f"The {noun} was asked a question",
        lambda noun: f"The {noun} eats daily",
    ],
    "Collective": [
        lambda noun: f"This {noun} consists of many members",
        lambda noun: f"The {noun} gathered together",
    ],
    # not very reliable
    # "Concrete/Abstract": [
    #     lambda noun: f"I can see a {noun}",
    #     lambda noun: f"You can touch the {noun}",
    # ],
    "Common": [lambda noun: f"The {noun} is useful", lambda noun: f"Any {noun} will do"],
    "Count": [
        lambda noun: f"One {noun}, two {inflect.plural_noun(noun)}",
        lambda noun: f"Several {inflect.plural_noun(noun)} were counted",
    ],
    "Mass": [
        lambda noun: f"Some {noun} was left",
        lambda noun: f"How many {noun} are there",
    ],
}

TestConfig = TypedDict("TestConfig", {"positive": bool, "heuristic": Optional[HeuristicFunction]})

# Configuration for whether passing each test should add or subtract points
TEST_CONFIG: Dict[str, TestConfig] = {
    # "Agentive": {"positive": True, "heuristic": is_agentive},
    "Eventive": {"positive": True, "heuristic": None},
    "Pluralia Tantum": {"positive": True, "heuristic": None},
    "Animate/Inanimate": {"positive": True, "heuristic": None},
    "Collective": {"positive": True, "heuristic": None},
    "Common": {"positive": True, "heuristic": None},
    "Count": {"positive": True, "heuristic": None},
    "Mass": {"positive": True, "heuristic": None},
}


def calculate_score(
    prediction: Dict[str, float], test_config: TestConfig, noun: Optional[str] = None
) -> float:
    # Use heuristic if available
    if test_config["heuristic"] is not None and noun is not None:
        return 0.8 if test_config["heuristic"](noun) else -0.8

    score = prediction["score"]
    if prediction["label"] == "LABEL_1":  # Test passed
        return score if test_config["positive"] else -score
    else:  # Test failed
        return -score if test_config["positive"] else score


def main():
    nouns = get_nouns("nounlist.txt")
    classifier = pipeline("text-classification", model="textattack/roberta-base-CoLA")

    # Create empty DataFrame with appropriate columns
    columns = ["noun"]
    for category in TESTS.keys():
        for i in range(len(TESTS[category])):
            columns.append(f"{category}_{i+1}")
    columns.append("total_score")  # Add column for total score

    df = pd.DataFrame(columns=columns)

    for noun in tqdm(nouns, desc="Processing nouns"):
        entry = {"noun": noun}
        category_scores = {}  # Store average scores for each category

        for label, test_functions in TESTS.items():
            category_score = 0.0
            for i, test_func in enumerate(test_functions):
                sentence = test_func(noun)
                prediction = classifier(sentence)[0]
                score = calculate_score(prediction, TEST_CONFIG[label], noun)
                category_score += score
                entry[f"{label}_{i+1}"] = f"{score:.2f}"

            # Calculate average score for this category
            category_scores[label] = category_score / len(test_functions)

        # Calculate total score as average of category scores
        total_score = sum(category_scores.values()) / len(category_scores)
        entry["total_score"] = f"{total_score:.2f}"

        # Append the new entry to the DataFrame
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    save_results(df)


def save_results(df: pd.DataFrame):
    # Save to CSV
    df.to_csv("noun_classification_results.csv", index=False)
    print("Done! Results saved to noun_classification_results.csv")

    # Convert total_score to float for sorting
    df["total_score"] = df["total_score"].astype(float)

    # Sort by total_score
    sorted_df = df.sort_values("total_score", ascending=False)

    # Display top and bottom results
    print("\nTop 10 Results:")
    print(sorted_df[["noun", "total_score"]].head(10).to_string(index=False))

    print("\nBottom 10 Results:")
    print(sorted_df[["noun", "total_score"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
