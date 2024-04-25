from utils.DataGenerator import pad, generateData
from utils.model_helper import DEVICE, SBERT_VERSION, MAX_SENT_LENGTH, MAX_PARA_LENGTH, HP_SBERT_AHEADS, TRANS_DROPOUT, HP_SBERT_LAYERS, HP_LR_SBERT
from utils.model_helper import MENU, SAVE_HISTORY, SAVE_MODEL, HP_SBERT_HIDDEN, EMB_SIZE, BATCH_SIZE, N_EPOCH
from utils.ModelScore import ProduceAUC, plot_loss
import numpy as np
from tqdm import tqdm
from transformers import AutoModel
import torch
from torch import nn
from models.TransformerModel import TransformerModel


def train(model, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir, prev_ep_val_loss = 100):
    num_epoch = N_EPOCH
    patience = 100
    earlystop_cnt = 0

    for epoch in range(num_epoch):
        train_epoch_loss = 0.0
        val_epoch_loss = 0.0
        instance_cnt = 0
        for ids, ids_b, label, id in tqdm(train_generator):
            pad(ids, MAX_PARA_LENGTH, MAX_SENT_LENGTH)     
            pad(ids_b, MAX_PARA_LENGTH, MAX_SENT_LENGTH)      

            idst = torch.as_tensor(ids).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
            ids_bt = torch.as_tensor(ids_b).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
            
            with torch.no_grad():
                emb = encoder(idst).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
                emb_b = encoder(ids_bt).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)

            y_pred = model(emb, emb_b).to(DEVICE)   
            y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
            train_loss = criterion(y_pred, y_true)

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            train_epoch_loss += y_pred.shape[0] * train_loss.item()
            instance_cnt += len(id)

        train_epoch_loss /= instance_cnt
        history['train loss'].append(train_epoch_loss)


        instance_cnt = 0
        for ids, ids_b, label, id in tqdm(val_generator):
            pad(ids, MAX_PARA_LENGTH, MAX_SENT_LENGTH)
            pad(ids_b, MAX_PARA_LENGTH, MAX_SENT_LENGTH)

            idst = torch.as_tensor(ids).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
            ids_bt = torch.as_tensor(ids_b).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
            
            with torch.no_grad():
                emb = encoder(idst).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
                emb_b = encoder(ids_bt).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)

                y_pred = model(emb, emb_b).to(DEVICE)   
                y_true = torch.as_tensor(label, dtype = torch.float32).to(DEVICE)
                val_loss = criterion(y_pred, y_true)

            val_epoch_loss += y_pred.shape[0] * val_loss.item()
            instance_cnt += len(id)

        val_epoch_loss /= instance_cnt
        history['val loss'].append(val_epoch_loss)
        print(f'epoch: {epoch}, training loss = {train_epoch_loss:.4f}, validation loss = {val_epoch_loss:.4f}')
        SAVE_HISTORY(history, hist_dir)


        if val_epoch_loss < prev_ep_val_loss:
            print(f'Improved from previous epoch ({prev_ep_val_loss:.4f}), model checkpoint saved to {model_dir}.')
            earlystop_cnt = 0
            SAVE_MODEL(model, optimizer, model_dir, val_epoch_loss)
            prev_ep_val_loss = val_epoch_loss
        else:
            if earlystop_cnt < patience: 
                print(f'No improvement from previous epoch ({prev_ep_val_loss:.4f})')
                earlystop_cnt += 1
            else:
                print(f'No improvement from previous epoch ({prev_ep_val_loss:.4f})')
                break
        

# def eval(model, encoder, test_generator):
#     score_df = torch.load('score.pt')
#     record = input('Enter new record name:')
#     score_df[record] = np.nan

#     for ids, ids_b, label, id in tqdm(test_generator):
#         pad(ids, MAX_PARA_LENGTH, MAX_SENT_LENGTH)
#         pad(ids_b, MAX_PARA_LENGTH, MAX_SENT_LENGTH)

#         idst = torch.as_tensor(ids).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
#         ids_bt = torch.as_tensor(ids_b).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
        
#         with torch.no_grad():
#             emb = encoder(idst).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
#             emb_b = encoder(ids_bt).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)

#             y_pred = model(emb, emb_b).cpu()

#         for i in range(len(id)):
#             score_df[record][id[i]] = y_pred.detach().numpy()[i]

#     torch.save(score_df, 'score.pt')
#     ProduceAUC()


from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, mean_squared_error, confusion_matrix
from scipy.spatial.distance import cosine
import numpy as np

def eval(model, encoder, test_generator):
    score_df = torch.load('score.pt')
    record = input('Enter new record name:')
    score_df[record] = np.nan

    y_true = []
    y_pred = []

    for ids, ids_b, label, id in tqdm(test_generator):
        pad(ids, MAX_PARA_LENGTH, MAX_SENT_LENGTH)
        pad(ids_b, MAX_PARA_LENGTH, MAX_SENT_LENGTH)

        idst = torch.as_tensor(ids).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
        ids_bt = torch.as_tensor(ids_b).view(BATCH_SIZE * MAX_PARA_LENGTH, -1).to(DEVICE)
        
        with torch.no_grad():
            emb = encoder(idst).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)
            emb_b = encoder(ids_bt).last_hidden_state.view(-1, MAX_PARA_LENGTH, MAX_SENT_LENGTH, EMB_SIZE)

            y_pred_batch = model(emb, emb_b).cpu()

        y_true.extend(label)
        y_pred.extend(y_pred_batch.detach().numpy())

        for i in range(len(id)):
            score_df[record][id[i]] = y_pred_batch.detach().numpy()[i]

    torch.save(score_df, 'score.pt')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    # Calculate and display metrics
    f1 = f1_score(y_true, y_pred > 0.5)
    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred > 0.5)
    rec = recall_score(y_true, y_pred > 0.5)
    cos_sim = 1 - cosine(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Cosine Similarity: {cos_sim:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    ProduceAUC()


if __name__ == "__main__":
    train_generator, val_generator, test_generator = generateData(BATCH_SIZE)
    encoder = AutoModel.from_pretrained(SBERT_VERSION).to(DEVICE)

    option, model_dir, hist_dir = MENU()

    config = {"emb_size": EMB_SIZE, 
              "max_n_sent": MAX_PARA_LENGTH, 
              "n_hidden": HP_SBERT_HIDDEN, 
              "HP_SBERT_AHEADS": HP_SBERT_AHEADS, 
              "n_layers": HP_SBERT_LAYERS, 
              "dropout": TRANS_DROPOUT}

    transformer = TransformerModel(**config).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr = HP_LR_SBERT)

    if option == '1':    
        history = {'train loss':[], 'val loss':[]}
        train(transformer, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir)
        plot_loss(history)
    
    elif option == '2':   
        checkpoint = torch.load(model_dir)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = torch.load(hist_dir)
        val_loss = checkpoint['validation_loss']
        transformer.train()
        train(transformer, encoder, criterion, optimizer, train_generator, val_generator, history, model_dir, hist_dir, val_loss)
        plot_loss(history)
    

    else:    
        checkpoint = torch.load(model_dir)
        transformer.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss = checkpoint['validation_loss']

        transformer.eval()
        eval(transformer, encoder, test_generator)
