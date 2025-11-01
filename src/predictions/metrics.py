"""Module metrics.py"""
import json
import os
import typing

import numpy as np
import pandas as pd

import config
import src.elements.structures as st
import src.functions.objects


class Metrics:
    """
    Metrics
    """

    def __init__(self):
        pass

    @staticmethod
    def __metrics(data: pd.DataFrame, stage: typing.Literal['training', 'testing']) -> dict:
        """

        :param data: A frame of measures, estimates, and errors
        :param stage: The data of the training or testing stage
        :return:
        """

        _se: np.ndarray = np.power(data['ae'].to_numpy(), 2)
        _r_mse: float = np.sqrt(_se.mean())

        return {'rmse': float(_r_mse),
                'mape': float(data['ape'].mean()),
                'mae': float(data['ae'].mean()),
                'stage': stage}

    def exc(self, structures: st.Structures) -> list[dict]:
        """

        :param structures: An object of data frames vis-à-vis training & testing estimates, etc.
        :return:
        """

        return [self.__metrics(data=structures.training, stage='training'),
                self.__metrics(data=structures.training, stage='testing')]
