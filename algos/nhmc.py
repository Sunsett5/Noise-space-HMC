import torch
import math
def hmc(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig):
    tau = opt.tau
    epsilon = opt.epsilon
    L = max(1,math.floor(tau/epsilon))
    epochs = 40
    sampling = 20

    orig_pic = []
    for j in range(len(x_orig)):
        orig_pic.append(inverse_data_transform(config, x_orig[j]))
    psnr_list = []
    loss_list = []
    final_img_list = []
    sigma_y_list = []

    d = x.view(-1).numel()
    r = math.sqrt(d)
    x = x / (torch.sum(x**2)**0.5) * r
    x = x.detach().requires_grad_()

    accepted = 0
    rejected = 0

    epoch = 0
    """ for k in range(100):
        xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
        loss = torch.sum((y_0 - H_funcs.H(xt))**2)
        loss_grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
        x = sphere_update(x, loss_grad, r, alpha=3/(k+1)**0.4).detach().requires_grad_()
        x_save = [inverse_data_transform(config, y) for y in xt.detach()]
        for j in range(len(x_save)):
            tvu.save_image(
                x_save[j], os.path.join(opt.image_folder, f"hmc_{epoch}.png")
            )
            mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            #psnr_list.append(psnr.item())
            print('k', k, 'PSNR:', psnr.item(), 'ratio', (torch.sum(x**2)**0.5).item() / r) """

    while epoch < epochs + 2 * sampling:
        if epoch < epochs:
            sigma_y = opt.sigma_0 + 0.9 * (1 - epoch / epochs) ** 2
        elif epoch == epochs:
            sigma_y = opt.sigma_0
            if tau > 0.1:
                tau = 0.1
                epsilon = 0.01    

        # initialize momentum
        p = torch.randn_like(x, device=device) * math.sqrt(opt.m)
        xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
        loss = torch.sum((y_0 - H_funcs.H(xt))**2)
        current_loss = loss.detach()
        loss_grad = torch.autograd.grad(loss, x, retain_graph=False)[0]

        H = (1/2) * torch.sum(x**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * current_loss + (1/2)* torch.sum(p * p, dim=(1, 2, 3)) * opt.m**(-1)

        x_proposal = x.detach().clone().requires_grad_(True)

        # update momentum
        p = p - (epsilon / 2) * (x_proposal.detach() + 1/(2 * sigma_y**2) * loss_grad)

        for l in range(L):

            x_proposal = x_proposal + epsilon * opt.m**(-1) * p 
            x_proposal = x_proposal.detach().requires_grad_(True)

            xt = iterative_sampling(x_proposal, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
            loss = torch.sum((y_0 - H_funcs.H(xt))**2)
            loss_grad = torch.autograd.grad(loss, x_proposal, retain_graph=False)[0]

            p = p - epsilon * (x_proposal.detach() + 1/(2 * sigma_y**2) * loss_grad)

        p = p + (epsilon / 2) * (x_proposal.detach() + 1/(2 * sigma_y**2) * loss_grad)
        proposal_loss = loss.detach()

        H_proposal = (1/2) * torch.sum(x_proposal**2, dim=(1, 2, 3)) + (1/(2 * sigma_y**2)) * proposal_loss + (1/2)* torch.sum(p * p, dim=(1, 2, 3)) * opt.m**(-1)
        delta_H = H_proposal - H
        print('prob_ratio', torch.exp((current_loss-proposal_loss)/(2*sigma_y**2)).mean().item())
        acceptance_ratio = min(torch.tensor([1], device=device), torch.exp(-delta_H))
        accept = torch.rand(1).item() < acceptance_ratio.item()
        if accept:
            accepted += 1
            rejected = 0
            x_accept = xt.detach().clone()
            if epoch >= epochs + sampling:
                final_img_list.append(x_accept[0])
            epoch += 1
            
            x = x_proposal.detach().clone().requires_grad_(True)

            x_save = [inverse_data_transform(config, y) for y in xt.detach()]
            for j in range(len(x_save)):
                tvu.save_image(
                    x_save[j], os.path.join(opt.image_folder, f"hmc_{epoch}.png")
                )
                mse = torch.mean((x_save[j].to(device) - orig_pic[j]) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                psnr_list.append(psnr.item())
                sigma_y_list.append(sigma_y)
                print('epoch', epoch, 'PSNR:', psnr.item(), 'sigma_y:', sigma_y, 'tau:', tau, 'ratio', (torch.sum(x**2)**0.5).item() / r)
        else:
            rejected += 1
            if rejected >= 2:
                tau = tau * 0.95
                print('                    Rejected too many times, annealing tau:', tau)
                epsilon = epsilon * 0.95
            continue

    """ skip = 0
    # plot the PSNR, loss in the same graph
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(psnr_list[skip:], 'g-', label='PSNR')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('PSNR', color='g')

    ax2 = ax1.twinx()
    ax2.plot(sigma_y_list[skip:], 'b-', label='sigma_y')
    ax2.set_ylabel('sigma_y', color='b')

    # save
    ax1.set_title('HMC Sampling: PSNR, sigma_y over Epochs')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)


    # Optional: Combine legends
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.savefig(os.path.join(opt.image_folder, 'hmc_combined.png'), bbox_inches='tight') """

    return torch.stack(final_img_list) #x_accept