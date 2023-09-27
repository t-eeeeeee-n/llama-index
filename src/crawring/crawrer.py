import csv, os
from urllib.parse import urlparse
from gpt_model import llama

def scrape_lamaindex(target: str, csv_file_name: str):
    """
    Llama-indexを使用したスクレイピング
    """
    print(f"Scraping {target}...")
    # TODO URL一覧をどこに置いてどこから取得するか
    # csvを配列に変換
    url_list = []
    with open(csv_file_name) as f:
        reader = csv.reader(f)
        for row in reader:
            url_list.append(row[0])

    # インデックス作成
    llama.create_index(url_list, target)

    # 会社名
    company_name = "DYM"
    llama.exec_query(target, company_name)

def main_process():
    # Define root domain to crawl
    full_url = "https://dym.asia/about/"

    llama.crawl(full_url)

    local_domain = urlparse(full_url).netloc
    url_directory = os.path.join("./data", "url/")
    csv_file_name = f"{url_directory}{local_domain}_urls.csv"
    scrape_lamaindex(local_domain, csv_file_name)

