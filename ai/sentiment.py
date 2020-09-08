from tensorflow.keras.preprocessing.sequence import pad_sequences


class SentimentCalculator:
    def __init__(self):
        self.SENTIMENT_THRESHOLDS = (0.4, 0.7)

    def _clean_text(self, text):
        return text.replace("'", "")

    def _calculate_sentiment(self, score):
        label = 'NEUTRAL'
        if score <= self.SENTIMENT_THRESHOLDS[0]:
            label = 'NEGATIVE'
        elif score >= self.SENTIMENT_THRESHOLDS[1]:
            label = 'POSITIVE'

        return label

    def predict(self, text, model, tokenizer):
        text = self._clean_text(text)
        x = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
        score = model.predict([x])[0]
        label = self._calculate_sentiment(score)

        return {'label': label, 'score': float(score)}
