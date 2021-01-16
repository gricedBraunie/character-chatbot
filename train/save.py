from src.savestorage import DropboxLoader
from dotenv import load_dotenv
load_dotenv()
import os
import click
import logging

fmt='%(asctime)s - %(message)s'
datefmt='%d-%b-%y %H:%M:%S'
logging.basicConfig(filename='savedata.log', format=fmt, level=logging.INFO, datefmt=datefmt)

@click.command()
@click.argument('model', required=True, type=click.File('rb'))
@click.argument('intentfile', required=True, type=click.File('rb'))
def main(model, intentfile):
    """
    Saves the data on your storage service (currently Dropbox)
    """
    TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN")
    ACCESS_KEY = os.environ.get("DROPBOX_APP_KEY")

    dbl = DropboxLoader(TOKEN, ACCESS_KEY)
    dbl.log_on()

    logging.info("Saving model")
    try:
        dbl.save_file(model, 'model/chatbot_model')
    except:
        logging.error(f"Wrong data file, type: {type(model)}")

    logging.info("Saving data")
    try:
        dbl.save_file(intentfile, 'data/intents.json')
    except:
        logging.error(f"Wrong data file, type: {type(intentfile)}")




if __name__=="__main__":
    main()