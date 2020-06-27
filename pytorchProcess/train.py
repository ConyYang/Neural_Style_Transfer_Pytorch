from pytorchProcess.optimizer import get_input_param_optimier




def run_style_transfer(content_img, style_img, input_img, num_epochs,
                       model, content_loss_list, style_loss_list):
    print("Building the style transfer model")
    input_param, optimizer = get_input_param_optimier(input_img)
    print("Optimizing")
    for epoch in range(num_epochs):
        def closure():
            input_param.data.clamp_(0, 1)
            # Input G into model to get output of each network layer
            model(input_param)
            style_score = 0
            content_score = 0
            # clear
            optimizer.zero_grad()
            # calculate total loss and gradient of each loss
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            # print every time
            if epoch % 1 == 0:
                print('run {}/10'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.data.item(), content_score.data.item()))
                print()

            return style_score + content_score
        optimizer.step(closure)

    return input_param.data



