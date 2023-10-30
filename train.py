# --encoding:utf-8--
# author:jiayawei
"""
train the model
"""
import torch
import os


def train_model_with_adamw(model, dataloader, criterion, optimizer, scheduler,
                           num_epochs=5, validation_dataloader=None, patience=3,
                           logger=None, model_path=None):
    best_validation_loss = float('inf')
    patience_counter = 0
    # check whether the model_path exists, if not exists, create it
    if model_path is not None:
        if not os.path.exists(model_path):
            os.makedirs(model_path)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for input1, input2, labels in dataloader:
            optimizer.zero_grad()
            logits = model(input1, input2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        if logger is None:
            print(f"Epoch {epoch + 1}, Loss: {average_loss}")
        else:
            logger.info(f"Epoch {epoch + 1}, Loss: {average_loss}")

        if validation_dataloader is not None:
            validation_loss = evaluate_model(model, validation_dataloader, criterion)
            if logger is None:
                print(f"Validation Loss: {validation_loss}")
            else:
                logger.info(f"Validation Loss: {validation_loss}")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), model_path +'/' +  'siamese_bert_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if logger is None:
                    print(f"Early stopping after {epoch + 1} epochs.")
                else:
                    logger.info(f"Early stopping after {epoch + 1} epochs.")
                break


def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for input1, input2, labels in dataloader:
            logits = model(input1, input2)
            loss = criterion(logits, labels)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    model.train()
    return average_loss
