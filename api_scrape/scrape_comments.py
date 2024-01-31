from dataclasses import dataclass
from datetime import datetime
from io import StringIO
import os
import re
from typing import List

import boto3
import pandas as pd
import requests


@dataclass
class CommentData:
    """Comment Data class"""

    id: str
    type: str
    links: str
    # insdie 'attributes'
    commentOnDocumentId: str
    duplicateComments: str
    address1: str
    address2: str
    agencyId: str
    city: str
    category: str
    comment: str
    clean_comment: str
    job: str
    country: str
    docAbstract: str
    docketId: str
    documentType: str
    email: str
    fax: str
    firstName: str
    lastName: str
    field1: str
    field2: str
    fileFormats: str
    govAgency: str
    govAgencyType: str
    objectId: str
    legacyId: str
    modifyDate: str
    organization: str
    originalDocumentId: str
    pageCount: str
    phone: str
    postedDate: str
    postmarkDate: str
    reasonWithdrawn: str
    receiveDate: str
    restrictReason: str
    restrictReasonType: str
    stateProvinceRegion: str
    submitterRep: str
    submitterRepAddress: str
    submitterRepCityState: str
    subtype: str
    title: str
    trackingNbr: str
    withdrawn: str
    zip: str
    openForComment: str
    # in included
    attachments: str
    attachment_link: str

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)


def nlp_clean(str_to_clean: str) -> str:
    """
    cleans strings for use with nlp services

    """
    clean_str = re.sub(
        r"<[^>]+>",
        " ",
        str_to_clean,
    )
    clean_str = re.sub(
        r"&quot;",
        "",
        clean_str,
    )
    clean_str = re.sub(
        r"&#39;",
        "",
        clean_str,
    )
    clean_str = re.sub(
        r"-",
        " ",
        clean_str,
    )
    clean_str = re.sub(
        r"[^a-zA-Z0-9 \.!,]",
        "",
        clean_str,
    )
    string_encode = clean_str.encode("ascii", "ignore")
    string_decode = string_encode.decode()

    return string_decode


def get_job(comment: str) -> str:
    job_re = re.compile("[Aa]s a [A-Z a-z]+,|M\\.?D\\.?|Dr\\.")
    job = job_re.findall(comment)
    if len(job) == 0:
        job = ""
    else:
        job = job[0]
        job = re.sub(r"[Aa]s a  ?", "", job)
        job = re.sub(r",", "", job)
        job = re.sub(r"\.", "", job)
        if job == "Dr":
            job = "Doctor"
        elif job == "MD":
            job = "MD"
        else:
            job = job.title()
    return job


def get_json(url: str) -> dict:
    response = requests.get(url)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def get_comment_data(
    API_KEY_1: str, urls: List[str], comment_data: List[CommentData]
) -> List[CommentData]:
    for url in urls:
        full_url = f"{url}?include=attachments&api_key={API_KEY_1}"
        # get raw comment data
        comment_json = get_json(full_url)

        # check if there are attachments
        if len(comment_json) > 1:
            attachments = 1
            attachment_link = comment_json["included"][0]["attributes"]["fileFormats"][
                0
            ]["fileUrl"]
        else:
            attachments = 0
            attachment_link = None

        # clean comment
        clean_comment = nlp_clean(comment_json["data"]["attributes"]["comment"])
        job_title = get_job(clean_comment)

        # save comment data
        new_comment = CommentData(
            id=comment_json["data"]["id"],
            type=comment_json["data"]["type"],
            links=url,
            commentOnDocumentId=comment_json["data"]["attributes"][
                "commentOnDocumentId"
            ],
            duplicateComments=comment_json["data"]["attributes"]["duplicateComments"],
            address1=comment_json["data"]["attributes"]["address1"],
            address2=comment_json["data"]["attributes"]["address2"],
            agencyId=comment_json["data"]["attributes"]["agencyId"],
            city=comment_json["data"]["attributes"]["city"],
            category=comment_json["data"]["attributes"]["category"],
            comment=comment_json["data"]["attributes"]["comment"],
            country=comment_json["data"]["attributes"]["country"],
            docAbstract=comment_json["data"]["attributes"]["docAbstract"],
            docketId=comment_json["data"]["attributes"]["docketId"],
            documentType=comment_json["data"]["attributes"]["documentType"],
            email=comment_json["data"]["attributes"]["email"],
            fax=comment_json["data"]["attributes"]["fax"],
            firstName=comment_json["data"]["attributes"]["firstName"],
            lastName=comment_json["data"]["attributes"]["lastName"],
            field1=comment_json["data"]["attributes"]["field1"],
            field2=comment_json["data"]["attributes"]["field2"],
            fileFormats=comment_json["data"]["attributes"]["fileFormats"],
            govAgency=comment_json["data"]["attributes"]["govAgency"],
            govAgencyType=comment_json["data"]["attributes"]["govAgencyType"],
            objectId=comment_json["data"]["attributes"]["objectId"],
            legacyId=comment_json["data"]["attributes"]["legacyId"],
            modifyDate=comment_json["data"]["attributes"]["modifyDate"],
            organization=comment_json["data"]["attributes"]["organization"],
            originalDocumentId=comment_json["data"]["attributes"]["originalDocumentId"],
            pageCount=comment_json["data"]["attributes"]["pageCount"],
            phone=comment_json["data"]["attributes"]["phone"],
            postedDate=comment_json["data"]["attributes"]["postedDate"],
            postmarkDate=comment_json["data"]["attributes"]["postmarkDate"],
            reasonWithdrawn=comment_json["data"]["attributes"]["reasonWithdrawn"],
            receiveDate=comment_json["data"]["attributes"]["receiveDate"],
            restrictReason=comment_json["data"]["attributes"]["restrictReason"],
            restrictReasonType=comment_json["data"]["attributes"]["restrictReasonType"],
            stateProvinceRegion=comment_json["data"]["attributes"][
                "stateProvinceRegion"
            ],
            submitterRep=comment_json["data"]["attributes"]["submitterRep"],
            submitterRepAddress=comment_json["data"]["attributes"][
                "submitterRepAddress"
            ],
            submitterRepCityState=comment_json["data"]["attributes"][
                "submitterRepCityState"
            ],
            subtype=comment_json["data"]["attributes"]["subtype"],
            title=comment_json["data"]["attributes"]["title"],
            trackingNbr=comment_json["data"]["attributes"]["trackingNbr"],
            withdrawn=comment_json["data"]["attributes"]["withdrawn"],
            zip=comment_json["data"]["attributes"]["zip"],
            openForComment=comment_json["data"]["attributes"]["openForComment"],
            attachments=attachments,
            attachment_link=attachment_link,
            clean_comment=clean_comment,
            job=job_title,
        )
        # add comment to list of comments
        comment_data.append(new_comment)

    return comment_data


# save df to s3 bucket
def lambda_handler(event, context):

    FILE_NAME = "COMM_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".csv"
    BUCKET = "cart-raw-data"  # already created on S3

    # sample urls
    urls = [
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0004",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0005",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0006",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0007",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0008",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0009",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0010",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0011",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0012",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0013",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0014",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-0015",
        "https://api.regulations.gov/v4/comments/VA-2020-VHA-0024-2317",
    ]

    # cycle through URLs to get all comment data
    comments = []
    comments = get_comment_data(event["API_KEY_1"], urls, comments)
    comments_df = pd.DataFrame(comments)
    print(comments_df.info())

    # init s3 bucket
    s3 = boto3.client("s3")
    csv_buffer = StringIO()
    comments_df.to_csv(csv_buffer)
    s3.put_object(Bucket=BUCKET, Key=FILE_NAME, Body=csv_buffer.getvalue())

    # local testing
    # comments_df.to_csv(FILE_NAME, index=False)
    print("done")
