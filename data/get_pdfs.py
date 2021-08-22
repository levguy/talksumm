'''
Retrieves the pdf files from the URLs provided in 
/data/talksumm_papers_urls.txt
Author: Tomas Goldsack
'''

import urllib.request
from time import sleep
import numpy as np
import re, logging, os, requests
from datetime import datetime

ACL_BASE_URL = "https://aclanthology.org/"
ARXIV_BASE_URL = "https://arxiv.org/pdf/"

if not os.path.exists("./logs"):
  os.makedirs("./logs")

if not os.path.exists("./pdf"):
  os.makedirs("./pdf")
  downloaded_titles = []
else:
  downloaded_titles = [fpath.split("/")[-1].split(".")[0] for fpath in os.listdir("./pdf")]

logging.basicConfig(
  filename="./logs/get_pdfs.log.{}".format(datetime.timestamp(datetime.now())), 
  level=logging.INFO,
  format = '%(asctime)s | %(levelname)s | %(message)s'
)

def download_file(download_url, filename):
  '''Downloads and saves the pdf file'''
  response = urllib.request.urlopen(download_url)    
  file = open(filename + ".pdf", 'wb')
  file.write(response.read())
  file.close()


def get_pdf_links(response):
  '''Retrieves PDF URLs from response.text'''
  regex = re.compile('(http)(?!.*(http))(.*?)(\.pdf)')
  matches = list(set(["".join(link) for link in regex.findall(response.text)]))  
  return matches  


def multiple_links_handler(pdf_urls):
  '''Gets PDF URLs for specific sites where the generic method finds more than 1 URL'''
  pdf_urls = [link for link in pdf_urls if "-supp" not in link]

  if len(pdf_urls) > 1:
    springer_urls = [link for link in pdf_urls if "link.springer.com" in link]
    pdf_urls = springer_urls if len(springer_urls) > 0 else pdf_urls

  return pdf_urls


def no_links_handler(response, url):
  '''Gets PDF URLs for specific sites where the generic method finds no URLs'''

  if "aclweb" in url or "aclweb" in response.url:
    # Retrieve ACL code from original URL (uppercase)
    idx = -2 if url.endswith("/") else -1
    acl_code = url.split("/")[idx].upper()
    # PDF URL format for ACL papers 
    return [ACL_BASE_URL + acl_code + ".pdf"]
  
  if "arxiv" in url or "arxiv" in response.url:
    idx = -2 if url.endswith("/") else -1
    arxiv_code = url.split("/")[idx]
    # PDF URL format for ARXIV papers 
    return [ARXIV_BASE_URL + arxiv_code + ".pdf"]
  
  if "openreview" in url or "openreview" in response.url:
    openrv_url =  url if "openreview" in url else response.url
    return [openrv_url.replace("forum", "pdf")]
  
  if "transacl" in url or "transacl" in response.url:
    tacl_url =  url if "transacl" in url else response.url
    tacl_regex = re.compile('(http)(?!.*(http))(.*?)(\/tacl\/article\/view\/[[0-9]+\/[[0-9]+)')
    view_urls = list(set(["".join(link) for link in tacl_regex.findall(response.text)]))
    view_urls = [link for link in view_urls if not link.endswith("/0")]
    
    if len(view_urls) > 1:
      # Has more than 1 full text link, take the one with the lowest suffix id number 
      # (this is consistenly the desired TACL site URL)
      suffixes = [int(link.split("/")[-1]) for link in view_urls]
      min_ind = suffixes.index(min(suffixes))
      view_urls = [view_urls[min_ind]]

    if (len(view_urls) == 1):
      return [view_urls[0].replace("view", "download")]
    else:
      return view_urls

  if ("iaaa" in url or "iaaa" in response.url) or (
    "aaai" in url or "aaai" in response.url
  ):
    iaaa_url =  url if ("iaaa" in url or "iaaa" in url) else response.url
    iaaa_regex = re.compile('(http)(?!.*(http))(.*?)(\/paper\/view\/[[0-9]+\/[[0-9]+)')
    view_urls = list(set(["".join(link) for link in tacl_regex.findall(response.text)]))
    return view_urls

  if "mdpi" in url or "mdpi" in response.url:
    mdpi_url =  url if "mdpi" in url else response.url
    mdpi_regex = re.compile('(.*?)(\/[[0-9]+\/[[0-9]+\/[[0-9]+\/pdf)')
    view_urls = list(set(["".join(link) for link in tacl_regex.findall(response.text)]))
    return view_urls

  if "ceur-ws" in url or "ceur-ws" in response.url:
    ceur_url =  url if "ceur-ws" in url else response.url
    idx = -2 if url.endswith("/") else -1
    return [ceur_url + ceur_url.split("/")[idx] + ".pdf"] 

  if "isca-speech" in url or "isca-speech" in response.url:
    isca_url =  url if "isca-speech" in url else response.url
    isca_url = isca_url.replace("abstracts", "pdfs")
    return [isca_url.replace(".html", ".PDF")]

  return []  

failed_titles = []
with open("./talksumm_papers_urls.txt", "r") as input_txt:
  for line in input_txt.readlines():
    try:
      title, url = line.rstrip().split("\t") 
      logging.info(f'Processing "{title}"')

      if title in downloaded_titles:
        logging.info(f'Already downloaded "{title}"')
        continue

      # Sleep to prevent connection reset 
      sleep(np.random.randint(1, 10))

      # Make request to given URL
      response = requests.get(url, allow_redirects=True)

      #Retrieve URLs to PDFs from response
      pdf_links = get_pdf_links(response)

      # Handle too many/too few links
      if len(pdf_links) > 1:
        logging.warning(f'Too many PDF URLs found: {pdf_links}')
        pdf_links = multiple_links_handler(pdf_links)

      if len(pdf_links) < 1:
        logging.warning(f'No PDF links found on "{url}"')
        pdf_links = no_links_handler(response, url)

      if len(pdf_links) == 1:
        logging.info(f'Got a single PDF URL: "{pdf_links[0]}"')
        download_url = pdf_links[0]
      elif url.endswith(".pdf"): # three provided URLs are PDF links
        download_url = url
      else:
        failed_titles.append((title, url))
        raise Exception(f'Got {len(pdf_links)} PDF URLs ({pdf_links})')

      # Download PDF
      download_file(download_url, "./pdf/" + title)
      logging.info(f'Successfully retrieved PDF')
    
    except Exception as e:
      logging.error(f'Failed to retrieve PDF: {e}')

logging.info('Finished processing paper titles')
logging.warning(f'Failed to retrieve {len(failed_titles)} PDFs')
logging.warning(f'Failed titles and urls: {failed_titles}')