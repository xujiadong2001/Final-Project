
import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

seq2seq_list = ['seq2seq_gru', 'seq2seq_gru_attention', 'seq2seq_transformer']
def l1_regularization(model, lamda):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lamda * l1_norm

def train_model_w_metrics(
    model_type,
    model,
    label_encoder,
    train_generator,
    val_generator,
    learning_params,
    save_dir,
    error_plotter=None,
    calculate_train_metrics=False,
    device='cpu',
    return_result = False,
):
    # tensorboard writer for tracking vars
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard_runs'))

    train_loader = torch.utils.data.DataLoader(
        train_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    val_loader = torch.utils.data.DataLoader(
        val_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    # define loss
    loss = nn.MSELoss()
    # loss = nn.L1Loss()

    # define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_params['lr'],
        betas=(learning_params["adam_b1"], learning_params["adam_b2"]),
        weight_decay=learning_params['adam_decay']
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=learning_params['lr_factor'],
        patience=learning_params['lr_patience'],
        min_lr=1e-7,
        verbose=True,
    )

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # for convenience
    target_label_names = label_encoder.target_label_names

    def run_epoch(loader, n_batches, training=True):

        epoch_batch_loss = []
        epoch_batch_acc = []

        if not training or calculate_train_metrics:
            # complete dateframe of predictions and targets
            acc_df = pd.DataFrame(columns=[*target_label_names, 'overall_acc'])
            err_df = pd.DataFrame(columns=target_label_names)
            pred_df = pd.DataFrame(columns=target_label_names)
            targ_df = pd.DataFrame(columns=target_label_names)

        for batch in loader:

            # get inputs
            inputs, labels_dict = batch['images'], batch['labels']

            # wrap them in a Variable object
            inputs = Variable(inputs).float().to(device)

            # get labels
            labels = label_encoder.encode_label(labels_dict)

            # set the parameter gradients to zero
            if training:
                optimizer.zero_grad()
                if model_type in seq2seq_list:
                    labels = labels.permute(0, 2, 1)  # [batch_size, timesteps, out_dim]
                    outputs_tmp = model(inputs, output_last=False, target=labels)
                    # 合并batch_size和timesteps
                    outputs = outputs_tmp.view(-1, outputs_tmp.size(-1))
                    # labels shape [batch_size, out_dim,timesteps]
                    labels = labels.contiguous().view(-1, labels.size(-1))
                    # labels = labels.view(-1, labels.size(-1))
                else:
                    outputs = model(inputs)


            # forward pass, backward pass, optimize
            else:
                with torch.no_grad():
                    model.eval()
                    if model_type in seq2seq_list:
                        labels = labels.permute(0, 2, 1)  # [batch_size, timesteps, out_dim]
                        outputs_tmp = model(inputs, output_last=False)
                        outputs = outputs_tmp.reshape(-1, outputs_tmp.size(-1))

                        labels = labels.contiguous().view(-1, labels.size(-1))
                        # labels = labels.view(-1, labels.size(-1))
                    else:
                        outputs = model(inputs)
                    model.train()

            loss_size = loss(outputs, labels)
            epoch_batch_loss.append(loss_size.item())

            if training:
                # l1 regularization
                if learning_params['l1_reg'] > 0.0:
                    loss_size += l1_regularization(model, learning_params['l1_reg'])
                loss_size.backward()
                optimizer.step()

            # calculate metrics that are useful to keep track of during training
            # this can slow learning noticably, particularly if train metrics are tracked
            if not training or calculate_train_metrics:

                # decode predictions into label
                if model_type in seq2seq_list: # [batch_size, timesteps, out_dim]
                    outputs = outputs_tmp[:, -1, :] # [batch_size, out_dim]
                    # label shape [batch_size, timesteps]
                    # 取最后一个timestep的label
                    labels_dict = {k: v[:, -1] for k, v in labels_dict.items()}

                predictions_dict = label_encoder.decode_label(outputs)

                # append predictions and labels to dataframes
                batch_pred_df = pd.DataFrame.from_dict(predictions_dict)
                batch_targ_df = pd.DataFrame.from_dict(labels_dict)
                pred_df = pd.concat([pred_df, batch_pred_df])
                targ_df = pd.concat([targ_df, batch_targ_df])

                # get metrics useful for training
                batch_err_df, batch_acc_df = label_encoder.calc_batch_metrics(labels_dict, predictions_dict)

                # append error to dataframe
                err_df = pd.concat([err_df, batch_err_df])
                acc_df = pd.concat([acc_df, batch_acc_df])

                # statistics
                epoch_batch_acc.append(acc_df['overall_acc'].mean())
            else:
                epoch_batch_acc.append(0.0)

        if not training or calculate_train_metrics:
            # reset indices to be 0 -> test set size
            acc_df = acc_df.reset_index(drop=True).fillna(0.0)
            err_df = err_df.reset_index(drop=True).fillna(0.0)
            pred_df = pred_df.reset_index(drop=True).fillna(0.0)
            targ_df = targ_df.reset_index(drop=True).fillna(0.0)
            return epoch_batch_loss, epoch_batch_acc, acc_df, err_df, pred_df, targ_df
        else:
            return epoch_batch_loss, epoch_batch_acc

    # get time for printing
    training_start_time = time.time()

    # for tracking metrics across training
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # for saving best model
    lowest_val_loss = np.inf
    best_model = None
    time_start = time.time()
    with tqdm(total=learning_params['epochs']) as pbar:

        # Main training loop
        for epoch in range(1, learning_params['epochs'] + 1):

            if not calculate_train_metrics:
                train_epoch_loss, train_epoch_acc = run_epoch(
                    train_loader, n_train_batches, training=True
                )
            else:
                train_epoch_loss, train_epoch_acc, train_acc_df, train_err_df, _, _ = run_epoch(
                    train_loader, n_train_batches, training=True
                )

            # ========= Validation =========
            model.eval()
            val_epoch_loss, val_epoch_acc, val_acc_df, val_err_df, val_pred_df, val_targ_df = run_epoch(
                val_loader, n_val_batches, training=False
            )
            model.train()

            # append loss and acc
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            # print metrics
            print("")
            print("")
            print("Epoch: {}".format(epoch))
            print("")

            print("Train Metrics")

            print("train_err: {:.6f}".format(np.mean(train_epoch_loss)))
            if calculate_train_metrics:
                print(train_err_df[target_label_names].mean())

            print("train_acc: {:.6f}".format(np.mean(train_epoch_acc)))
            if calculate_train_metrics:
                print(train_acc_df[target_label_names].mean())

            print("")
            print("Validation Metrics")
            print("val_err: {:.6f}".format(np.mean(val_epoch_loss)))
            print(val_err_df[target_label_names].mean())

            print("val_acc: {:.6f}".format(np.mean(val_epoch_acc)))
            print(val_acc_df[target_label_names].mean())
            print("")

            # write vals to tensorboard
            writer.add_scalar('loss/train', np.mean(train_epoch_loss), epoch)
            writer.add_scalar('loss/val', np.mean(val_epoch_loss), epoch)
            writer.add_scalar('accuracy/train', np.mean(train_epoch_acc), epoch)
            writer.add_scalar('accuracy/val', np.mean(val_epoch_acc), epoch)
            writer.add_scalar('learning_rate', get_lr(optimizer), epoch)

            for label_name in target_label_names:
                if calculate_train_metrics:
                    writer.add_scalar(f'accuracy/train/{label_name}', train_acc_df[label_name].mean(), epoch)
                    writer.add_scalar(f'loss/train/{label_name}', train_err_df[label_name].mean(), epoch)
                writer.add_scalar(f'accuracy/val/{label_name}', val_acc_df[label_name].mean(), epoch)
                writer.add_scalar(f'loss/val/{label_name}', val_err_df[label_name].mean(), epoch)

            '''
            # track weights on tensorboard
            for name, weight in model.named_parameters():
                full_name = f'{os.path.basename(os.path.normpath(save_dir))}/{name}'
                writer.add_histogram(full_name, weight, epoch)
                writer.add_histogram(f'{full_name}.grad', weight.grad, epoch)
            '''

            # update plots
            if error_plotter:
                if error_plotter.plot_during_training:
                    error_plotter.update(
                        val_pred_df, val_targ_df, val_err_df
                    )

            # save the model with lowest val loss
            if np.mean(val_epoch_loss) < lowest_val_loss:
                lowest_val_loss = np.mean(val_epoch_loss)

                print('Saving Best Model')
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, 'best_model.pth')
                )

                # save loss and acc, save val
                save_vars = [train_loss, val_loss, train_acc, val_acc]
                with open(os.path.join(save_dir, 'train_val_loss_acc.pkl'), 'bw') as f:
                    pickle.dump(save_vars, f)

                save_vars = [val_pred_df, val_targ_df, val_err_df, target_label_names]
                with open(os.path.join(save_dir, 'val_pred_targ_err.pkl'), 'bw') as f:
                    pickle.dump(save_vars, f)

            # decay the lr
            lr_scheduler.step(np.mean(val_epoch_loss))

            # update epoch progress bar
            pbar.update(1)

    print("Training finished, took {:.6f}s".format(time.time() - training_start_time))

    # save final model
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, 'final_model.pth')
    )




if __name__ == "__main__":
    pass
