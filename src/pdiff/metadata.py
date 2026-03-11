"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

import shutil
import pandas as pd
from pathlib import Path
import tifffile
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Union, Callable
import torch


class pDiffMetadata:
    """Tracks metadata (e.g. treatments, image file paths, and profiles) associated with training or generated data for pdiff.

    Attributes:
        df: the backing dataframe. Contains all metadata, especially treatment {names, profiles, image paths}.
    """

    image_string = "image_ch"
    profile_string = "profile"
    treatment_string = "treatment"
    treatment_index_string = "treatment_index"
    mask_string = "mask"
    extracted_image_fingerprint_string = "image_fingerprint_vector"

    def __init__(self, df_path: Union[Path, str]):
        df_path = Path(df_path)
        self.df_path = df_path
        self.df_parent_path = df_path.parent
        self.load(df_path)

    @staticmethod
    def initialize_dataframe(df_path: Path) -> pd.DataFrame:
        assert df_path.suffix == ".pkl"
        df_path.parent.mkdir(exist_ok=True, parents=True)
        multi_index = pd.MultiIndex(
            levels=[[], []],
            codes=[[], []],
            names=[
                pDiffMetadata.treatment_string,
                pDiffMetadata.treatment_index_string,
            ],
        )
        new_df = pd.DataFrame(
            index=multi_index,
            columns=[pDiffMetadata.profile_string, pDiffMetadata.image_string],
        )
        new_df.to_pickle(df_path)

    def add_image_data(
        self,
        output_root_path: Path,
        add_image_list: List[Union[np.ndarray, Image.Image]],
        treatment: Any,
        profile: np.ndarray,
    ) -> None:
        """add images and corresponding metadata to this metadata object, and save images

        Args:
            output_root_path (Path): root path to save images
            images (List[Union[np.ndarray, PIL.Image]]): List of images to add
            treatment (Any): treatment identifier
            profile (np.ndarray): treatment profile

        Returns:
            None
        """
        self.df[pDiffMetadata.profile_string] = self.df[
            pDiffMetadata.profile_string
        ].astype("object")
        for ii, image in enumerate(add_image_list):
            image_filename = "{}_{}.tiff".format(treatment, str(ii))
            image_path = Path(output_root_path / treatment / image_filename)
            image_path.parent.mkdir(exist_ok=True, parents=True)
            tifffile.imwrite(image_path, np.array(image))
            row_index = (treatment, ii)
            self.df.loc[row_index, pDiffMetadata.image_string] = image_path
            self.df.at[row_index, pDiffMetadata.profile_string] = profile
        self.save()

    @staticmethod
    def prepare_metadata_from_external(
        input_pickled_df_path: Path,
        treatment_column_name: str,
        profile_column_name: str,
        image_column_name_list: List[str],
        profile_length=2048,
        output_metadata_pickle_filepath: Path = Path("./prepared_metadata.pkl"),
    ) -> pd.DataFrame:
        """Format an input CSV to have standardized columns and column names for pDiff operations.

        Args:
            input_pickled_df_path: Path,
            treatment_column_name: str,
            profile_column_name: str,
            image_column_name_list: List[str],
            profile_length=2048,
            output_metadata_pickle_filepath: Path = Path("./prepared_metadata.pkl"),


        Returns:
            A DataFrame formatted with standardized column names for index (treatment, treatment id) and columns (profile, [image_ch0...image_chN]) and other columns dropped.

        """

        def add_indices_to_group(group):
            """adds column of sequential numerical indices to a group"""
            group[pDiffMetadata.treatment_index_string] = range(0, len(group))
            return group

        def build_rename_image_column_dict(
            image_column_name_list: List[str],
        ) -> Dict[str, str]:
            """build dictionary to convert the ordered list of image column names to standardized format.

            Standardized format is _image_string + i), i.e. image_ch0, image_ch1, etc.

            Args:
                image_column_name_list (List[str]): ordered list of column names to rename

            Returns:
                Dict[str, str]: mapping from old name to new standardized name
            """
            image_column_rename_dict = {}
            for ii, column_name in enumerate(image_column_name_list):
                new_column_name = pDiffMetadata.image_string + str(ii)
                image_column_rename_dict[column_name] = new_column_name
            return image_column_rename_dict

        df: pd.DataFrame = pd.read_pickle(input_pickled_df_path)
        df.reset_index(inplace=True)
        groupby = df.groupby(treatment_column_name)
        df = groupby.apply(add_indices_to_group, pDiffMetadata.treatment_index_string)
        df.reset_index(inplace=True, drop=True)
        columns_to_keep = [
            pDiffMetadata.treatment_index_string,
            treatment_column_name,
            profile_column_name,
            *image_column_name_list,
        ]
        columns_to_drop = set(df.columns) - set(columns_to_keep)
        df.drop(columns_to_drop, axis=1, inplace=True)
        df.rename(
            columns={
                profile_column_name: pDiffMetadata.profile_string,
                treatment_column_name: pDiffMetadata.treatment_string,
            },
            inplace=True,
        )
        rename_image_column_dict = build_rename_image_column_dict(
            image_column_name_list
        )
        df.rename(columns=rename_image_column_dict, inplace=True)
        df.set_index(
            [pDiffMetadata.treatment_string, pDiffMetadata.treatment_index_string],
            inplace=True,
        )

        df[pDiffMetadata.profile_string].apply(lambda x: x[:profile_length])
        df.to_pickle(output_metadata_pickle_filepath)
        return df

    @staticmethod
    def _image_to_uint8(image: np.ndarray) -> np.ndarray:
        """
        helper function to convert from higher bit representations e.g. uint16 to the expected uint8.
        """
        return (255 * (image / np.iinfo(image.dtype).max)).astype(np.uint8)

    def _modify_paths_helper(
        self,
        row: pd.Series,
        new_root_path: Path,
        path_levels_to_keep: int,
        do_copy: bool = False,
    ) -> pd.Series:
        """
        copy images plus folder paths (levels_to_copy) from end of current image paths to a new root directory and update corresponding metadata entries
        """
        for channel_column_name in self.image_channels:
            src_path = self.resolve_image_path(row[channel_column_name])
            src_path_parts = src_path.parts
            dest_path = new_root_path / Path(*src_path_parts[-path_levels_to_keep:])
            row[channel_column_name] = dest_path
            if do_copy:
                dest_path.parent.mkdir(exist_ok=True, parents=True)
                print(src_path, " to ", dest_path)
                shutil.copy(src_path, dest_path)
        return row

    def modify_image_paths(
        self, new_root_path: Path, path_levels_to_keep: int, do_copy: bool
    ) -> None:
        """copy all images and some levels of the current image directory paths to a new root directory, update metadata entries, and save updated metadata at root of new directory.

        Useful to copy a training subset from a large archive.
        """
        self.df = self.df.apply(
            self._modify_paths_helper,
            axis=1,
            new_root_path=new_root_path,
            path_levels_to_keep=path_levels_to_keep,
            do_copy=do_copy,
        )
        self.save(new_root_path / "metadata.pkl")

    def _truncate_image_path_helper(
        self,
        row: pd.Series,
        image_channel_columns: List[str],
        levels_to_keep: int,
    ) -> pd.Series:
        for channel_column_name in self.image_channels:
            src_path = Path(row[channel_column_name])
            src_path_parts = src_path.parts
            truncated_path = Path(*src_path_parts[-levels_to_keep:])
            row[channel_column_name] = truncated_path
        return row

    def truncate_image_paths(self, levels_to_keep: int) -> None:
        self.df = self.df.apply(
            self._truncate_image_path_helper,
            axis=1,
            image_channel_columns=self.image_channels,
            levels_to_keep=levels_to_keep,
        )

    def _set_image_channels(self):
        image_channel_list = list(
            self.df.filter(like=pDiffMetadata.image_string).columns
        )
        image_channel_list.sort()
        self.image_channels = image_channel_list

    def get_image(self, idx: int) -> Image.Image:
        """return the idx-th image from the metadata.
        Supports multiple images per channel (e.g. separate channel images stored individually)
        Args:
            idx (int): index

        Returns:
            Image: idx-th image, combined from multiple channels if necessary.
        """
        channel_image_list = []
        for channel_image_path in self.get_image_paths(idx):
            assert channel_image_path.is_file()
            channel_image = tifffile.imread(channel_image_path)
            channel_image = self._image_to_uint8(channel_image)
            channel_image_list.append(channel_image)
        if len(channel_image_list) > 1:
            multichannel_image = np.stack(channel_image_list, axis=-1)
        else:
            multichannel_image = channel_image_list[0]
        return Image.fromarray(multichannel_image)

    def get_all_images(
        self, image_transform: Callable[[np.ndarray], np.ndarray] = None
    ) -> List[Image.Image]:
        if image_transform is None:
            return [self.get_image(x) for x in range(len(self))]
        else:
            return [
                self.get_image_with_transform(x, image_transform)
                for x in range(len(self))
            ]

    def get_mask(self, idx: int) -> np.ndarray:
        """return the idx-th mask from the metadata.
        Args:
            idx (int): index

        Returns:
            Image: idx-th mask
        """
        assert self.mask_string in self.df.columns
        idx_series = self.df.iloc[idx]
        mask = tifffile.imread(idx_series[self.mask_string])
        return mask

    def get_extracted_image_fingerprint(self, idx: int) -> np.ndarray:
        """return the idx-th mask from the metadata.
        Args:
            idx (int): index

        Returns:
            Image: idx-th mask
        """
        assert self.extracted_image_fingerprint_string in self.df.columns
        idx_series = self.df.iloc[idx]
        image_fingerprint = np.array(
            idx_series[self.extracted_image_fingerprint_string]
        )
        return image_fingerprint

    def get_image_with_transform(
        self, idx: int, image_transform: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """get transformed image as ndarray

        Args:
            idx (int): _description_
            image_transform (function): _description_

        Returns:
            np.ndarray: idx-th image with transform applied, as image
        """
        # image = np.array(self.get_image(idx))
        image = self.get_image(idx)
        return image_transform(image)

    def get_profile(self, idx: int) -> np.ndarray:
        """
        return the idx-th profile as a 2D tensor of shape [1, profile_length]
        """
        idx_series = self.df.iloc[idx]
        profile = idx_series[self.profile_string]
        profile = np.reshape(profile, (1, -1))
        return profile

    def resolve_image_path(self, path_to_resolve: Union[Path, str]):
        resolved_path = Path(path_to_resolve)
        if not resolved_path.is_file():
            resolved_path = self.df_parent_path / resolved_path
            assert resolved_path.is_file()
        return resolved_path

    def get_image_paths(self, idx: int) -> List[Path]:
        idx_series = self.df.iloc[idx]
        return [
            self.resolve_image_path(idx_series[channel_column_name])
            for channel_column_name in self.image_channels
        ]

    def get_treatment_name(self, idx: int) -> str:
        """
        return the idx-th treatment name
        """
        return self.df.index.get_level_values(level=self.treatment_string)[idx]

    def __len__(self):
        return len(self.df)

    def get_unique_treatments_df(self) -> pd.DataFrame:
        """
        return a dataframe containing a single {treatment, profile, image} element for each treatment
        """
        return self.df.groupby(level=0).head(1)

    def get_treatment_dict(self) -> Dict[Any, np.array]:
        """
        returns a dict of {treatment : profile}
        """
        unique_treatments_df = self.get_unique_treatments_df()
        unique_treatment_keys = list(unique_treatments_df.index.unique(level=0))
        unique_profile_values = list(unique_treatments_df[self.profile_string])
        return dict(zip(unique_treatment_keys, unique_profile_values))

    def save(self, output_file_path: Path = None) -> None:
        """save current configuration metadata as a pickled dataframe, loadable by another pDiffMetadata object.
        The newly saved object is now treated as the "current" for any modifying operations.
        """
        if output_file_path is None:
            output_file_path = self.df_path
        output_file_path.parent.mkdir(exist_ok=True, parents=True)
        self.df.to_pickle(output_file_path)
        self.df_path = output_file_path

    def load(self, load_dataframe_path: Path) -> None:
        """
        load configuration metadata from a pickled dataframe
        """
        assert load_dataframe_path.is_file()
        df = pd.read_pickle(load_dataframe_path)
        assert len(df.index.names) == 2
        assert self.profile_string in df.columns
        assert self.treatment_string in df.index.names
        assert self.treatment_index_string in df.index.names

        self.df = df
        self.df_path = load_dataframe_path
        self.df_parent_path = load_dataframe_path.parent
        self._set_image_channels()

    def apply_cellpose(
        self,
        cellpose_mask_output_path: Path,
        cellpose_params: Dict[str, Any],
        image_transform: Callable[[np.ndarray], np.ndarray] = None,
        device=torch.device("cpu"),
    ) -> None:
        from pdiff.analysis import init_cellpose, run_cellpose

        cellpose_model = init_cellpose()
        cellpose_mask_output_path.mkdir(exist_ok=True, parents=True)
        self.df[self.extracted_image_fingerprint_string] = 1
        self.df[self.extracted_image_fingerprint_string] = self.df[
            self.extracted_image_fingerprint_string
        ].astype("object")
        for i, row in enumerate(self.df.iterrows()):
            row_index, row_series = row
            treatment_string = str(row_index[0])
            treatment_idx_string = str(row_index[1])
            if image_transform is not None:
                image = self.get_image_with_transform(
                    i, image_transform=image_transform
                )
            else:
                image = self.get_image(i)
            mask, style = run_cellpose(image, cellpose_model, cellpose_params)
            mask_filename = "{}_mask_{}.tiff".format(
                treatment_string, treatment_idx_string
            )
            mask_path = Path(
                cellpose_mask_output_path / treatment_string / mask_filename
            )
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            tifffile.imwrite(mask_path, mask)
            self.df.at[row_index, self.mask_string] = mask_path
            self.df.at[row_index, self.extracted_image_fingerprint_string] = style
        self.save()
