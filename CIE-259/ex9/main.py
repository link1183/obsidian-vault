import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from typing import List, Dict, Any

API_KEY = "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"


class HuggingFaceSentimentAnalyzer:
    """
    A class to perform sentiment analysis using Hugging Face's free API.
    Includes visualization capabilities for the analyzed sentiments.
    """

    def __init__(self, api_token: str | None = None):
        """
        Initialize with your Hugging Face API token if you have one.
        Even without a token, you can make limited API calls.

        Args:
            api_token: Your Hugging Face API token (optional)
        """
        self.api_token = api_token
        self.headers = {"Content-Type": "application/json"}
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"

        # Default model for sentiment analysis
        self.model_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

        # Initialize tokenizer for fallback local processing
        self.tokenizer = None

    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze the sentiment of provided texts using Hugging Face API.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of sentiment analysis results for each text
        """

        results = []
        for text in texts:
            # Rate limiting to avoid overloading free API
            time.sleep(1)

            payload = {"inputs": text}

            try:
                response = requests.post(
                    self.model_url, headers=self.headers, json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    results.append(
                        {"text": text, "sentiment": result[0], "error": None}
                    )
                else:
                    results.append(
                        {
                            "text": text,
                            "sentiment": None,
                            "error": f"API Error: {response.status_code} - {response.text}",
                        }
                    )
            except Exception as e:
                results.append(
                    {"text": text, "sentiment": None, "error": f"Exception: {str(e)}"}
                )

        return results

    def visualize_sentiment(
        self,
        results: List[Dict[str, Any]],
        title: str = "Sentiment Analysis Results",
        save_path: str | None = None,
    ) -> None:
        """
        Visualize sentiment analysis results with a horizontal stacked bar chart.

        Args:
            results: List of sentiment analysis results
            title: Title for the visualization
            save_path: Path to save the visualization (optional)
        """
        # Filter out results with errors
        valid_results = [r for r in results if r["sentiment"] is not None]

        if not valid_results:
            print("No valid results to visualize!")
            return

        # Extract data for visualization
        texts = [
            r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"]
            for r in valid_results
        ]

        # Create a figure with a custom color gradient
        _, ax = plt.subplots(figsize=(12, len(texts) * 0.8 + 2))

        # Define custom colormap: red for negative, blue for positive
        colors = ["#FF5555", "#5555FF"]

        # Create data arrays
        data = []
        labels = []
        colors = []

        for result in valid_results:
            sentiment_data = result["sentiment"]
            for label_data in sentiment_data:
                label = label_data["label"]
                score = label_data["score"]

                # Normalize score for visualization
                data.append(score)
                labels.append(label)
                colors.append("red" if label == "NEGATIVE" else "blue")

        # Reshape data for visualization
        data_df = pd.DataFrame(
            {
                "text": texts,
                "NEGATIVE": [
                    (
                        r["sentiment"][1]["score"]
                        if r["sentiment"][1]["label"] == "NEGATIVE"
                        else r["sentiment"][0]["score"]
                    )
                    for r in valid_results
                ],
                "POSITIVE": [
                    (
                        r["sentiment"][0]["score"]
                        if r["sentiment"][0]["label"] == "POSITIVE"
                        else r["sentiment"][1]["score"]
                    )
                    for r in valid_results
                ],
            }
        )

        # Plot stacked horizontal bars
        sns.set_style("whitegrid")
        data_df.plot(
            x="text",
            y=["NEGATIVE", "POSITIVE"],
            kind="barh",
            stacked=True,
            color=["#FF5555", "#5555FF"],
            ax=ax,
            legend=True,
        )

        # Add percentage labels
        for i, _ in enumerate(texts):
            neg_score = data_df.iloc[i]["NEGATIVE"]
            pos_score = data_df.iloc[i]["POSITIVE"]

            # Add percentage labels
            ax.text(
                neg_score / 2,
                i,
                f"{neg_score:.1%}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
            ax.text(
                neg_score + pos_score / 2,
                i,
                f"{pos_score:.1%}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

        # Style the plot
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel("Sentiment Probability", fontsize=12)
        plt.ylabel("Text", fontsize=12)
        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def get_available_models(
        self, task: str = "sentiment-analysis"
    ) -> List[Dict[str, Any]]:
        """
        Get available models for a specific task from Hugging Face.

        Args:
            task: The task to search models for

        Returns:
            List of available models for the task
        """
        try:
            url = f"https://huggingface.co/api/models?task={task}&sort=downloads&direction=-1&limit=20"
            response = requests.get(url)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching models: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception while fetching models: {str(e)}")
            return []

    def change_model(self, model_id: str) -> None:
        """
        Change the model used for sentiment analysis.

        Args:
            model_id: The Hugging Face model ID to use
        """
        self.model_url = f"https://api-inference.huggingface.co/models/{model_id}"
        print(f"Model changed to: {model_id}")


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = HuggingFaceSentimentAnalyzer(api_token=API_KEY)

    # Test texts with different sentiments
    texts = [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This movie was terrible. I wouldn't recommend it to anyone.",
        "The weather is okay today, nothing special. The ending wasn't amazing, but it was good.",
        "I'm feeling very disappointed with the service provided.",
        "This is the best experience I've ever had with customer support!",
    ]

    # Analyze sentiments
    print("Analyzing sentiments...")
    results = analyzer.analyze_sentiment(texts)

    # Print results
    for result in results:
        if result["error"]:
            print(f"Error for '{result['text'][:30]}...': {result['error']}")
        else:
            sentiment = result["sentiment"]
            print(f"Text: '{result['text'][:50]}...'")
            print(
                f"Sentiment: {sentiment[0]['label']} ({sentiment[0]['score']:.2%}), "
                f"{sentiment[1]['label']} ({sentiment[1]['score']:.2%})"
            )
            print("-" * 50)

    # Visualize results
    analyzer.visualize_sentiment(results, title="Sentiment Analysis of Example Texts")

    # Get available models (optional)
    print("\nFetching available sentiment analysis models...")
    models = analyzer.get_available_models()
    print(f"Found {len(models)} models.")

    # Print top 5 models
    for i, model in enumerate(models[:5]):
        print(f"{i+1}. {model['id']} - Downloads: {model.get('downloads', 'N/A')}")
