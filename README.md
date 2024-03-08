# msds_capstone

### Capstone Project for MSDS 498

### Project Title: Automated Comment Analysis for Regulations.gov

The government is required to post proposed regulations on the website regulations.gov.  The public is invited to comment on the proposed regulations.  The government is required to read and consider all comments.  The government receives thousands of comments on each proposed regulation.  The comments are posted on the website in PDF format.  The comments are not searchable.  This solution aims to automate the process of reading and analyzing the comments.  The solution will use optical character recognition (OCR) to convert the PDFs to text.  The solution will use natural language processing (NLP) to analyze the comments.  The solution will use machine learning to classify the comments.  The solution will use a web application to display the results.

An example post is here: https://www.regulations.gov/comment/VA-2020-VHA-0024-9178

The list of all comments is found here: https://www.regulations.gov/docket/VA-2020-VHA-0024/comments

There are 13,324 comments on this proposed regulation. 


## To Run the Code

1. Clone the repository
2. Install the required packages
3. Run the code
4. Open the web application



This program was developed using [poetry](https://python-poetry.org/). After installing poetry by following their documentation, you will be able to reproduce this program on your computer with the following code (or you can do it the old fashioned way and install the packages manually):


#### Clone the Repository:

```bash
git clone https://github.com/papagorgio23/msds_capstone.git
cd msds_capstone
```




#### Prepare Enviornment:

``` bash
poetry shell
poetry install
```


#### Run Demo Application:


```bash
streamlit run app/app.py
```



