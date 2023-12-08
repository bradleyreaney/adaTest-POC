import os
import transformers
import adatest
from dotenv import load_dotenv

load_dotenv()

# create a HuggingFace sentiment analysis model
classifier = transformers.pipeline("sentiment-analysis", top_k=None)

# specify the backend generator used to help you write tests
generator = adatest.generators.AI21('j2-mid', api_key=os.environ.get('AI21_API_KEY'))

# Generate the output path if it doesn't exist
path = "./output"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)

# create a new test tree
tests = adatest.TestTree("./output/Nimble_Chat_Bot.csv")
# tests = adatest.TestTreeBrowser()

# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)
adatest.serve(tests.adapt(classifier, generator, auto_save=True))