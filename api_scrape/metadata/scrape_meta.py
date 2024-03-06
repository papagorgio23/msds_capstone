from dataclasses import dataclass
import datetime
import os
import re
import requests

API_KEY_1 = os.environ.get("API_KEY_1")
API_KEY_2 = os.environ.get("API_KEY_2")
COMMENT_ID = os.environ.get("COMMENT_ID")
i = 1
url = f"https://api.regulations.gov/v4/comments?filter[commentOnId]={COMMENT_ID}&page[size]=250&page[number]={i}&sort=lastModifiedDate,documentId&api_key={API_KEY_1}"


@dataclass
class CommentMeta:
    """Comment MetaData class"""

    id: str
    type: str
    # insdie 'attributes'
    documentType: str
    lastModifiedDate: datetime
    highlightedContent: int
    withdrawn: int
    title: int
    objectId: int
    postedDate: datetime
    # inside 'links' named 'self'
    link: str

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "documentType": self.documentType,
            "lastModifiedDate": self.lastModifiedDate,
            "highlightedContent": self.highlightedContent,
            "withdrawn": self.withdrawn,
            "title": self.title,
            "objectId": self.objectId,
            "postedDate": self.postedDate,
            "link": self.link,
        }


def clean_date(str_to_clean: str) -> str:
    """
    cleans date strings for use with API calls
    """
    clean_str = re.sub(
        "T",
        "%20",
        str_to_clean,
    )
    clean_str = re.sub(
        "Z",
        "",
        clean_str,
    )
    return clean_str


def get_json(url):
    response = requests.get(url)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def get_comment_meta_single_page(json, comment_meta):
    for comment in json["data"]:
        comment_meta.append(
            CommentMeta(
                id=comment["id"],
                type=comment["type"],
                documentType=comment["attributes"]["documentType"],
                lastModifiedDate=comment["attributes"]["lastModifiedDate"],
                highlightedContent=comment["attributes"]["highlightedContent"],
                withdrawn=comment["attributes"]["withdrawn"],
                title=comment["attributes"]["title"],
                objectId=comment["attributes"]["objectId"],
                postedDate=comment["attributes"]["postedDate"],
                link=comment["links"]["self"],
            )
        )
    return comment_meta


def get_comment_meta_20_pages(last_modified_date):
    all_comment_meta = []
    for i in range(1, 21):
        url = f"https://api.regulations.gov/v4/comments?filter[commentOnId]={COMMENT_ID}&filter[lastModifiedDate][ge]={last_modified_date}&page[size]=250&page[number]={i}&sort=lastModifiedDate,documentId&api_key={API_KEY_1}"
        print(url)
        json = get_json(url)
        all_comment_meta = get_comment_meta_single_page(json, all_comment_meta)
        # all_comment_meta.append(all_comment_meta)
        last_page = json["meta"]["lastPage"]
        if last_page:
            return all_comment_meta
        i += 1
    return all_comment_meta


def update_url(url):
    global i
    i += 1
    url = f"https://api.regulations.gov/v4/comments?filter[commentOnId]={COMMENT_ID}&page[size]=250&page[number]={i}&sort=lastModifiedDate,documentId&api_key={API_KEY_1}"
    return url


def update_url_date(last_modified_date):
    global i
    i += 1
    url = f"https://api.regulations.gov/v4/comments?filter[commentOnId]={COMMENT_ID}&filter[lastModifiedDate][ge]={last_modified_date}&page[size]=250&page[number]={i}&sort=lastModifiedDate,documentId&api_key={API_KEY_1}"
    return url


# get


def get_comment_meta_all():
    """
    get all comment meta data
    """
    all_comments = []
    for i in range(1, 21):
        last_modified_date = all_comments[-1].lastModifiedDate
        last_modified_date = clean_date(last_modified_date)
        print(last_modified_date)
        comments = get_comment_meta_20_pages(last_modified_date)
        all_comments + all_comments
        i += 1

    url = f"https://api.regulations.gov/v4/comments?filter[commentOnId]={COMMENT_ID}&page[size]=250&page[number]=1&sort=lastModifiedDate,documentId&api_key={API_KEY_1}"
    json = get_json(url)
    comment_meta = get_comment_meta(json)
    while json["meta"]["totalPages"] > 1:
        url = update_url(url)
        json = get_json(url)
        comment_meta += get_comment_meta(json)
    return comment_meta
