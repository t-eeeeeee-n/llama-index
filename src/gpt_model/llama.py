import csv, os, re
import urllib.request
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
from langchain.embeddings import OpenAIEmbeddings
from llama_index import GPTVectorStoreIndex, SimpleWebPageReader, LLMPredictor, PromptHelper, ServiceContext, \
    LangchainEmbedding, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

################################################################################
### Step 1
################################################################################

# Function to get the hyperlinks from a URL
def get_hyperlinks(url):
    
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            print(response)
            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

################################################################################
### Step 2
################################################################################

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.endswith((".pdf", ".doc", ".docx")):
                continue
            elif link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))

################################################################################
### Step 3
################################################################################

def crawl(url):
    print("Crawling " + url)
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the csv files
    if not os.path.exists("./data"):
            os.mkdir("./data")
    
    # Create a subdirectory named "index" inside the "data" directory
    index_directory = os.path.join("./data", "index/")
    url_directory = os.path.join("./data", "url/")

    if not os.path.exists(url_directory):
        os.makedirs(url_directory)
        
    if not os.path.exists(index_directory):
        os.makedirs(index_directory)

    # Create a CSV file to store the URLs
    csv_file_name = url_directory + local_domain + "_urls.csv"
    with open(csv_file_name, "w", newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

    # While the queue is not empty, continue crawling
    while queue:
        # Get the next URL from the queue
        url = queue.pop()
        print(url)  # for debugging and to see the progress

        # Append the URL to the CSV file
        with open(csv_file_name, "a", newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([url])

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

def create_index(url_list: list, target: str):
    """
    index作成

    Args；
        url_list (list)： URL一覧
        target (str)： 対象
    """
    print(f"Creating index for {target}...")
    # 使用するモデルの設定
    documents = SimpleWebPageReader(html_to_text=True).load_data(url_list)
    # TODO モデル検討
    # https://platform.openai.com/docs/models/overview
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    llama_embed = LangchainEmbedding(OpenAIEmbeddings(model="text-embedding-ada-002"), embed_batch_size=1)

    # Promptの設定
    prompt_helper = create_prompt_helper()

    # ServiceContextの作成
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        embed_model=llama_embed,
        prompt_helper=prompt_helper,
    )

    # Indexの作成
    index = GPTVectorStoreIndex.from_documents(documents=documents, service_context=service_context)

    # TODO 保存先検討
    # 保存
    index.storage_context.persist(persist_dir="./data/index/" + target)


def create_prompt_helper():
    """
    Promptの設定
    """
    print("Creating prompt...")
    # max LLM token input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    #
    chunk_overlap_ratio = 0.1
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, chunk_overlap_ratio, max_chunk_overlap)

    return prompt_helper

def exec_query(target: str, company_name: str):
    """
    クエリー実行

    Args：
        target (str)： 対象
        company_name (str)：会社名
    """
    print(f"Executing query for {target}...")
    PROMPT = """
        の
        以下の情報を提供してください：
        1. 会社名
        2. メールアドレス
        3. 電話番号
        4. 代表者名
        5. サービス内容
        これらの情報をJSON形式で提供してください：
        {
            "company": "会社名",
            "email": "メールアドレス",
            "phone": "電話番号",
            "representative": "代表者名",
            "service_description": "サービス内容",
        }
        情報が不足している場合は、ブランクの情報を提供してください。
    """
    PROMPT = f"{company_name}{PROMPT}"
    # TODO 複数ファイル行ける？
    # インデックスファイル読み込み
    storage_context = StorageContext.from_defaults(
        docstore=SimpleDocumentStore.from_persist_dir(persist_dir="./data/index/" + target),
        vector_store=SimpleVectorStore.from_persist_dir(persist_dir="./data/index/" + target),
        index_store=SimpleIndexStore.from_persist_dir(persist_dir="./data/index/" + target),
    )
    # don't need to specify index_id if there's only one index in storage context
    vector_store_index = load_index_from_storage(storage_context)

    # クエリー実行
    query_engine = vector_store_index.as_query_engine()
    response = query_engine.query(PROMPT)

    for i in response.response.split("。"):
        print(i + "。")
