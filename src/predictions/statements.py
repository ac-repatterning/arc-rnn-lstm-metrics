import typing

import numpy as np
import pandas as pd

import src.elements.specification as sc
import src.elements.structures as st


class Statements:

    def __init__(self):
        pass

    @staticmethod
    def __get_values(errors: pd.DataFrame, quantiles: pd.DataFrame, specification: sc.Specification,
                     stage: typing.Literal['training', 'testing']):

        _se: np.ndarray = np.power(errors['error'].to_numpy(), 2)
        _r_mean_se: float = np.sqrt(_se.mean())
        _r_median_se: float = np.sqrt(np.median(_se))

        _quantiles = quantiles.iloc[0, :].squeeze()

        instance = {
            'r_mean_se': float(_r_mean_se),
            'r_median_se': float(_r_median_se),
            'mean_pe': float(errors['p_error'].mean()),
            'median_pe': float(errors['p_error'].median()),
            'mean_e': float(errors['error'].mean()),
            'median_e': float(errors['error'].median()),
            'l_whisker': _quantiles.l_whisker,
            'u_whisker': _quantiles.u_whisker,
            'catchment_id': specification.catchment_id,
            'catchment_name': specification.catchment_name,
            'station_name': specification.station_name,
            'river_name': specification.river_name,
            'ts_id': specification.ts_id,
            'stage': stage}

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
