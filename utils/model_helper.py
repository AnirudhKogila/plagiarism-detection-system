import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SBERT_VERSION = "sentence-transformers/paraphrase-mpnet-base-v2"
MAX_SENT_LENGTH = 128
MAX_PARA_LENGTH = 8
BATCH_SIZE = 4
EMB_SIZE = 768

N_HIDDEN = 100
N_EPOCH = 20
HP_WINDOWS_SIZE = [2, 3]
HP_LR_CNN = 0.0001

HP_LR_SBERT = 0.0001
HP_SBERT_HIDDEN = 50
HP_SBERT_AHEADS = 2
HP_SBERT_LAYERS = 2
TRANS_DROPOUT = 0.2

POLY_M = 16
POLY_LR = 0.0001

TEST_PARAM = {'batch_size':BATCH_SIZE, 'shuffle': False}
TRAIN_PARAM = {'batch_size':BATCH_SIZE, 'shuffle': False}
VAL_PARAM = {'batch_size':BATCH_SIZE, 'shuffle': False}

def MENU():
    model_dir = ''
    hist_dir = ''
    print("Please select your option:")
    print("1. Train a new model.")
    print("2. Continue training the last model.")
    print("3. Evaluate the last model.")
    option = input('Your Option: ')
    model_dir = input('Model Directory:')
    if option == '3':
        return option, model_dir, hist_dir
    else:
        hist_dir = input('Training History Directory:')
        return option, model_dir, hist_dir

def SAVE_MODEL(mod, opt, dir, val_loss):
    torch.save({'model_state_dict': mod.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'validation_loss': val_loss}, dir)
    
def SAVE_HISTORY(his, dir):
    torch.save(his, dir)
