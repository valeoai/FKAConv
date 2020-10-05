# other imports
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import h5py

# torch imports
import torch
import torch.nn.functional as F
import torch.utils.data

from lightconvpoint.datasets.s3dis import S3DIS_Pillar_TrainVal as Dataset
import lightconvpoint.utils.metrics as metrics
from lightconvpoint.utils.network import get_conv, get_search
import lightconvpoint.utils.transformations as lcp_transfo
from lightconvpoint.utils.misc import wblue, wgreen

from fkaconv.networks.kpconv import KPConvSeg as Network
from fkaconv.networks.fusion import Fusion as NetworkFusion

# SACRED
from sacred import Experiment
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.config import save_config_file

SETTINGS.CAPTURE_MODE = "sys"  # for tqdm
ex = Experiment("S3DIS")
ex.captured_out_filter = apply_backspaces_and_linefeeds  # for tqdm
ex.add_config("config.yaml")
######


@ex.automain
def main(_run, _config):

    print(_config)

    savedir_root = _config['training']['savedir']
    device = torch.device(_config['misc']['device'])

    # save the config file
    os.makedirs(savedir_root, exist_ok=True)
    save_config_file(eval(str(_config)), os.path.join(
        savedir_root, "config.yaml"))
    
    # create the path to data
    rootdir = _config['dataset']['dir']

    N_CLASSES = 13

    # create the network
    print("Creating the network...", end="", flush=True)
    if "Fusion" == _config["network"]["model"]:
        def network_function():
            return NetworkFusion(
                3, N_CLASSES,
                get_conv(_config["network"]["backend_conv"]),
                get_search(_config["network"]["backend_search"]),
                config=_config
        )
    else:
        def network_function():
            return Network(
                3, N_CLASSES,
                get_conv(_config["network"]["backend_conv"]),
                get_search(_config["network"]["backend_search"]),
                config=_config
            )
    net = network_function()
    net.to(device)
    print("Done")

    training_transformations_data = [
        lcp_transfo.PillarSelection(_config["dataset"]["pillar_size"]),
        lcp_transfo.RandomSubSample(_config["dataset"]["num_points"])
    ]
    validation_transformations_data = [
        lcp_transfo.PillarSelection(_config["dataset"]["pillar_size"]),
        lcp_transfo.RandomSubSample(_config["dataset"]["num_points"])
    ]

    training_transformations_features=[
        lcp_transfo.ColorJittering(_config["training"]['jitter'])
    ]
    validation_transformations_features = []

    if not _config['training']['rgb']:
        training_transformations_features.append(lcp_transfo.NoColor())
        validation_transformations_features.append(lcp_transfo.NoColor())

    ds = Dataset(rootdir, _config, split='training', network_function=network_function,
            transformations_data=training_transformations_data,
            transformations_features=training_transformations_features)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=_config['training']['batch_size'], shuffle=True,
                                        num_workers=_config['misc']['threads']
                                        )

    ds_val = Dataset(rootdir, _config, split='validation', network_function=network_function,
            transformations_data=validation_transformations_data,
            transformations_features=validation_transformations_features)
    test_loader = torch.utils.data.DataLoader(ds_val, batch_size=_config['training']['batch_size'], shuffle=False,
                                        num_workers=_config['misc']['threads']
                                        )
    if _config['training']['weights']:
        weights= ds.get_class_weights().to(device)
    else:
        weights= torch.ones_like(ds.get_class_weights()).to(device)
    print("Done")


    print("Creating optimizer...", end="", flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=_config['training']['lr_start'])
    print("done")


    # iterate over epochs
    for epoch in range(0, _config['training']['epoch_nbr']):

        #######
        # training
        net.train()

        count=0

        train_loss = 0
        cm = np.zeros((N_CLASSES, N_CLASSES))
        t = tqdm(train_loader, ncols=100, desc="Epoch {}".format(epoch), disable=_config['misc']['disable_tqdm'])
        for data in t:

            pts = data['pts'].to(device)
            features = data['features'].to(device)
            seg = data['target'].to(device)
            net_ids = data["net_indices"]
            net_pts = data["net_support"]
            for i in range(len(net_ids)):
                net_ids[i] = net_ids[i].to(device)
            for i in range(len(net_pts)):
                net_pts[i] = net_pts[i].to(device)

            optimizer.zero_grad()
            outputs = net(features, pts, indices=net_ids, support_points=net_pts)
            loss =  F.cross_entropy(outputs, seg, weight=weights)
            loss.backward()
            optimizer.step()

            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
            cm += cm_

            oa = f"{metrics.stats_overall_accuracy(cm):.5f}"
            aa = f"{metrics.stats_accuracy_per_class(cm)[0]:.5f}"
            iou = f"{metrics.stats_iou_per_class(cm)[0]:.5f}"

            train_loss += loss.detach().cpu().item()

            t.set_postfix(OA=wblue(oa), AA=wblue(aa), IOU=wblue(iou), LOSS=wblue(f"{train_loss/cm.sum():.4e}"))

        ######
        ## validation
        net.eval()
        cm_test = np.zeros((N_CLASSES, N_CLASSES))
        test_loss = 0
        t = tqdm(test_loader, ncols=80, desc="  Test epoch {}".format(epoch), disable=_config['misc']['disable_tqdm'])
        with torch.no_grad():
            for data in t:

                pts = data['pts'].to(device)
                features = data['features'].to(device)
                seg = data['target'].to(device)
                net_ids = data["net_indices"]
                net_pts = data["net_support"]
                for i in range(len(net_ids)):
                    net_ids[i] = net_ids[i].to(device)
                for i in range(len(net_pts)):
                    net_pts[i] = net_pts[i].to(device)
                
                outputs = net(features, pts, indices=net_ids, support_points=net_pts)
                loss =  F.cross_entropy(outputs, seg)

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=1).copy()
                target_np = seg.cpu().numpy().copy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(N_CLASSES)))
                cm_test += cm_

                oa_val = f"{metrics.stats_overall_accuracy(cm_test):.5f}"
                aa_val = f"{metrics.stats_accuracy_per_class(cm_test)[0]:.5f}"
                iou_val = f"{metrics.stats_iou_per_class(cm_test)[0]:.5f}"

                test_loss += loss.detach().cpu().item()

                t.set_postfix(OA=wgreen(oa_val), AA=wgreen(aa_val), IOU=wgreen(iou_val), LOSS=wgreen(f"{test_loss/cm_test.sum():.4e}"))

        # create the root folder
        os.makedirs(savedir_root, exist_ok=True)

        # save the checkpoint
        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(savedir_root, "checkpoint.pth"))

        # write the logs
        logs = open(os.path.join(savedir_root, "logs.txt"), "a+")
        logs.write(f"{epoch} {oa} {aa} {iou} {oa_val} {aa_val} {iou_val}\n")
        logs.close()

        # log train values
        _run.log_scalar("trainOA", oa, epoch)
        _run.log_scalar("trainAA", aa, epoch)
        _run.log_scalar("trainIoU", iou, epoch)
        _run.log_scalar("testOA", oa_val, epoch)
        _run.log_scalar("testAA", aa_val, epoch)
        _run.log_scalar("testAIoU", iou_val, epoch)
