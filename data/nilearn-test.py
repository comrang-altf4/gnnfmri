from nilearn import datasets, plotting, image
from nilearn.plotting import plot_prob_atlas
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show

rest_dataset = datasets.fetch_development_fmri(n_subjects=10)

func_filenames = rest_dataset.func

from nilearn.decomposition import CanICA

canica = CanICA(
    n_components=20,
    memory="nilearn_cache",
    memory_level=2,
    verbose=10,
    mask_strategy="whole-brain-template",
    random_state=0,
    standardize="zscore_sample",
)
for j in range(10):
    canica.fit(func_filenames[j])

    # Retrieve the independent components in brain space. Directly
    # accessible through attribute `components_img_`.
    canica_components_img = canica.components_img_
    # components_img is a Nifti Image object, and can be saved to a file with
    # the following line:
    canica_components_img.to_filename("canica_resting_state.nii.gz")

    # Plot all ICA components together
    # plot_prob_atlas(canica_components_img, title="All ICA components")

    from nilearn.image import iter_img
    from nilearn.plotting import plot_stat_map, show

    for i, cur_img in enumerate(iter_img(canica_components_img)):
        plot_stat_map(
            cur_img,
            display_mode="z",
            # title=f"IC {int(i)}",
            cut_coords=1 ,
            colorbar=False,
            annotate=False,
            black_bg=True,
            symmetric_cbar=False,
            output_file=f"./data/output/{j}_IC_{int(i+1)}.png",
        )


    # plotting.show()