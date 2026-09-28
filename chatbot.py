from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pprint import pprint
from dotenv import load_dotenv

loader = DirectoryLoader(
    path="./papers",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
)

docs = loader.load()

MARKDOWN_SEPARATORS = [
    "\n#{1,6}",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n---+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap = 200,
    add_start_index = True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS
)

splits = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model=
)