import qr_lstm
from qr_lstm import BiDiLSTM
import preprocessing
import eval_qr

if __name__ == '__main__':

    # load model and reset optimizer and loss
    model, loss, opt = qr_lstm.initialize(new_model=True)

    for epoch in range(qr_lstm.EPOCHS):
        loss_this_epoch = qr_lstm.train_model_epoch(model, loss, opt)
        print("Epoch " + str(epoch) + " loss:" + str(loss_this_epoch))

        eval_qr.evaluate(model)
