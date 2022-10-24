class ModelEvaluator():
    def __init__(self, model):
        self.model = model

    def evaluate(self, data_loader):

        test_loss = 0.00
                            
        for num, batch in enumerate(data_loader):
                            
            predictions = self.model.forward(batch)
            batch_loss = self.model.loss((predictions, batch))
            
            test_loss = test_loss + batch_loss.detach()

        test_loss = test_loss / (num+1)

        return test_loss
