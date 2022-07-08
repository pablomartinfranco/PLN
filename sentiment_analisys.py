from happytransformer import HappyTextClassification
from happytransformer.happy_text_classification import TextClassificationResult
from dataclasses import dataclass
from typing import Callable
from pathlib import Path
from newsapi import NewsApiClient
from datetime import date, timedelta
from newspaper import Article
import nltk
import asyncio
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

nltk.data.path = [str(Path().resolve().parent.joinpath("nltk_data"))]


@dataclass(frozen=True)
class Digest:
    url: str
    html: str
    source: str
    authors: str
    publish_date: str
    keywords: str
    summary: str
    title: str
    text: str
    ners: str


@dataclass(frozen=True)
class Prediction:
    results: [(Digest, HappyTextClassification)]
    term: str


def process_article(article: Article) -> Digest:
    article.download()
    article.parse()
    article.nlp()
    text = " ".join(article.text.split()[:300])
    ners = get_ners(text)
    keywords = ", ".join(article.keywords)
    return Digest(
        url=article.url,
        html=article.html,
        source=article.source_url,
        authors=article.authors,
        publish_date=article.publish_date,
        keywords=keywords,
        summary=article.summary,
        title=article.title,
        text=text,
        ners=ners
    )


async def process_article_async(article: Article):
    return await asyncio.to_thread(process_article, article)


async def get_digests_async(term: str, sources: str, client: NewsApiClient,
                            from_param=date.today() - timedelta(days=1),
                            language="en"):

    response = client.get_everything(term,
                                     sources=sources,
                                     # category='business',
                                     from_param=from_param,
                                     language=language)

    digests = await asyncio.gather(
        *[process_article_async(Article(article["url"])) for article in response["articles"]]
    )

    return term, digests


async def get_digests_by_terms_async(terms: str, sources: str, client: NewsApiClient):
    return await asyncio.gather(
        *[get_digests_async(term, sources, client) for term in terms.split(',')]
    )


async def classify_sentiment_async(digest: Digest, classifier: HappyTextClassification):
    return digest, await asyncio.to_thread(classifier.classify_text, digest.text)


async def classify_digests_async(digests: [Digest], classifier: HappyTextClassification):
    return await asyncio.gather(
        *[classify_sentiment_async(digest, classifier) for digest in digests]
    )

def get_ners(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        if current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk


