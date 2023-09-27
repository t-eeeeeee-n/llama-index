import os
from crawring import crawrer

# Constants
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def main():
    crawrer.main_process()

if __name__ == "__main__":
    main()
