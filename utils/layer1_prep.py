import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = text.lower()  # Make text lowercase for consistency
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", '', text)  # Remove links, mentions, hashtags
    text = re.sub(r"[^a-z\s]", '', text)  # Remove numbers and punctuation
    from nltk.tokenize import RegexpTokenizer

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
 
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords + lemmatize
    return " ".join(tokens)  # Join back into a cleaned string

