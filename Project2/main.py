import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pc
from classifiers import (
    minimum_error, minimum_error_predict
)
from utils import (
    load_picture, extract_features
)

from PIL import Image
from config import PERM_DATA, HALL_DATA, WALL_DATA, FIGURES_PATH

# ---------------------------------------------------------------

def process(DATA, normalize=True):
    train_img = load_picture(DATA['train_path'], normalize=normalize)
    test_img = load_picture(DATA['test_path'], normalize=normalize)

    X_train, y_train = extract_features(train_img, DATA['regions'])
    tmp = 2 if normalize else 3
    X_out = train_img.reshape(-1, tmp)
    X_test = test_img.reshape(-1, tmp)

    params = minimum_error(X_train, y_train)
    test_result = minimum_error_predict(params, X_test).reshape(test_img.shape[:2])
    train_result = minimum_error_predict(params, X_out).reshape(train_img.shape[:2])

    return train_img, test_img, train_result, test_result, params


if __name__ == '__main__':

    DATA = PERM_DATA

    N_train_img, N_test_img, N_train_result, N_test_result, params = process(DATA)

    # ------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DATA['figx']*2, DATA['figy']))
    ax1.imshow(N_train_result, cmap='viridis')
    ax2.imshow(N_test_result, cmap='viridis')
    ax1.axis('off')
    ax2.axis('off')
    fig.suptitle("Normalised segmentation")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f'{DATA['name']}_norm_seg.pdf')


    # train_img, test_img, train_result, test_result = process(DATA, False)

    # ------------------------------------------------------------------

    train_img = load_picture(DATA['train_path'], False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DATA['figx']*2, DATA['figy']))
    ax1.imshow(train_img)
    for r in DATA['regions']:
        rect = pc.Rectangle((r.x1, r.y1), r.x2-r.x1, r.y2-r.y1, linewidth=2,
                            edgecolor='white', facecolor='none')
        ax1.add_patch(rect)
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_title('Segmented training image')
    ax2.set_title('Testing image')

    ax2.imshow(load_picture(DATA['test_path'], False))

    # fig.suptitle('Hallway images')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f'{DATA['name']}_patches.pdf')

    
    # ------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DATA['figx']*2, DATA['figy']))
    N_train_img = load_picture(DATA['train_path'], True)
    ax1.imshow(N_train_img[:, :, 0])
    ax2.imshow(N_train_img[:, :, 1])
    ax1.axis('off')
    ax2.axis('off')
    fig.suptitle('Normalized training image')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f'{DATA['name']}_norm_img.pdf')

    # ------------------------------------------------------------------
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    X_train, y_train = extract_features(N_train_img, DATA['regions'])
    axes[0].set_ylabel(f'normalized green')
    for idx, ax in enumerate(axes):
        X_k = X_train[y_train == idx]

        co = np.cov(X_k, rowvar=False, bias=True)
        eigval, eigvec = np.linalg.eigh(co)

        angle = np.rad2deg(np.arctan2(eigvec[1,1], eigvec[0, 1]))

        hb = ax.hexbin(X_k[:, 0], X_k[:, 1], bins='log', gridsize=50)
        ax.scatter(*X_k.mean(axis=0), label=f'mean', s=50, color='black')
        
        for i in [1, 2]:
            ellipse = pc.Ellipse(X_k.mean(axis=0), i*np.sqrt(eigval[1])*2, i*np.sqrt(eigval[0])*2, 
                       fc='none', ec='black', ls='--', angle=angle)
            ax.add_patch(ellipse)

        
        ax.set_xlabel(f'normalized red')
        ax.set_title(f'class: {idx}')
        # ax.set_aspect('equal')
        cb = fig.colorbar(hb, ax=ax, label='counts')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f'{DATA['name']}_test.pdf')


    
<<<<<<< HEAD
    # fig, ax = plt.subplots(figsize=(6,4))
    # X_train, y_train = extract_features(N_train_img, DATA['regions'])
    # # axes[0].set_ylabel(f'normalized red')
    # X = np.concatenate(N_train_img, axis=0)
    # ax.scatter(X[:, 0], X[:,1], alpha=0.01)

    # for idx in range(3):
    #     X_k = X_train[y_train == idx]

    #     ax.scatter(X_k[:, 0], X_k[:, 1], alpha=0.05)
    #     ax.scatter(*X_k.mean(axis=0), label=f'mean', s=50, color='black')
        
    #     # for i in [1, 2]:
    #     #     ellipse = pc.Ellipse(X_k.mean(axis=0), i*np.sqrt(eigval[1])*2, i*np.sqrt(eigval[0])*2, 
    #     #                fc='none', ec='black', ls='--', angle=angle)
    #     #     ax.add_patch(ellipse)

        
    #     ax.set_xlabel(f'normalized green')
    #     ax.set_title(f'class: {idx}')
    #     # ax.set_aspect('equal')
    # plt.tight_layout()
    # plt.savefig(FIGURES_PATH / f'{DATA['name']}_test2.pdf')
=======
    fig, ax = plt.subplots(figsize=(6,4))
    X_train, y_train = extract_features(N_train_img, DATA['regions'])
    # axes[0].set_ylabel(f'normalized red')
    X = np.concatenate(N_train_img, axis=0)
    ax.scatter(X[:, 0], X[:,1], alpha=0.01)

    for idx in range(3):
        X_k = X_train[y_train == idx]

        ax.scatter(X_k[:, 0], X_k[:, 1], alpha=0.05)
        ax.scatter(*X_k.mean(axis=0), label=f'mean', s=50, color='black')
        
        # for i in [1, 2]:
        #     ellipse = pc.Ellipse(X_k.mean(axis=0), i*np.sqrt(eigval[1])*2, i*np.sqrt(eigval[0])*2, 
        #                fc='none', ec='black', ls='--', angle=angle)
        #     ax.add_patch(ellipse)

        
        ax.set_xlabel(f'normalized green')
        ax.set_title(f'class: {idx}')
        # ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / f'{DATA['name']}_test2.pdf')
>>>>>>> refs/remotes/origin/main



    