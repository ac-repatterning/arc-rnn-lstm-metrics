"""Module src.predictions.interface.py"""
import logging

import dask

import config
import src.elements.master as mr
import src.elements.specification as sc
import src.functions.directories
import src.predictions.data


class Interface:
    """
    Interface
    """

    def __init__(self, arguments: dict):
        """

        :param arguments:
        """

        self.__arguments = arguments

        # Configurations
        self.__configurations = config.Config()

    @staticmethod
    def exc(specifications: list[sc.Specification]):
        """

        :param specifications:
        :return:
        """

        __get_data = dask.delayed(src.predictions.data.Data().exc)

        computations = []
        for specification in specifications:
            master: mr.Master = __get_data(specification=specification)
            computations.append(master.e_training.shape[0])

        messages = dask.compute(computations, scheduler='threads')[0]
        logging.info(messages)
