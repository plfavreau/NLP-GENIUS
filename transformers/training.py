import torch

def train(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0  # To calculate average loss per batch
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Ensure you are using outputs.loss to access the loss; outputs[0] also works but is less clear
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]  # More explicit handling
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
    
    # Optionally, print average loss per batch at the end of the epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss for Epoch {epoch+1}: {avg_loss:.4f}")