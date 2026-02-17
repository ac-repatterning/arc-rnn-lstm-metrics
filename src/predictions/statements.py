"""Module statements.py"""
import typing

import numpy as np
import pandas as pd

import src.elements.specification as sc
import src.elements.structures as st


class Statements:
    """
    This class outlines errors vis-à-vis a gauge station's river level prediction/forecasting model.
    """

    def __init__(self):
        pass

    @staticmethod
    def __get_values(errors: pd.DataFrame, quantiles: pd.DataFrame, specification: sc.Specification,
                     stage: typing.Literal['training', 'testing']) -> dict:
        """

        :param errors:
        :param quantiles:
        :param specification:
        :param stage:
        :return:
        """

        _se: np.ndarray = np.power(errors['error'].to_numpy(), 2)
        _r_mean_se: float = np.sqrt(_se.mean())
        _r_median_se: float = np.sqrt(np.median(_se))

        _quantiles: pd.Series = quantiles.iloc[0, :].squeeze()

        instance = {
            'r_mean_se': float(_r_mean_se),
            'r_median_se': float(_r_median_se),
            'mean_pe': float(errors['p_error'].mean()),
            'mean_e': float(errors['error'].mean()),
            'median_e': float(errors['error'].median()),
            'catchment_id': specification.catchment_id,
            'catchment_name': specification.catchment_name,
            'station_name': specification.station_name,
            'river_name': specification.river_name,
            'latitude': specification.latitude,
            'longitude': specification.longitude,
            'ts_id': specification.ts_id,
            'stage': stage}

        instance.update(_quantiles.to_dict())

        return instance

    def exc(self, structures: st.Structures, specification: sc.Specification) -> list[dict]:
        """

        :param structures: An object of data frames vis-à-vis training & testing estimates, etc.
        :param specification:
        :return:
        """

        return [
            self.__get_values(
                errors=structures.training, quantiles=structures.q_training, specification=specification, stage='training'),
            self.__get_values(
                errors=structures.testing, quantiles=structures.q_testing, specification=specification, stage='testing')]
