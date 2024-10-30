import typing

import torch
import torch.nn as nn

class DetectionModel(nn.Module):
    """
    ===========
    **Summary**
    ===========

        (borrame) Aqui un resumen de la clase

    ==============
    **Parameters**
    ==============

        (borrame) Aqui una descripcion de los parametros del constructor 

        * **nombre:** (*tipo*) Descripcion.
        * **nombre:** (*tipo*) Descripcion.

    ======================
    **Instance Variables**
    ======================

        (borrame) Aqui una descripcion de los atributos (publicos y privados) de la clase

         * **nombre:** (*tipo*) Descripcion.
         * **nombre:** (*tipo*) Descripcion.

    ===========
    **Example**
    ===========

        (borrame) Aqui un ejemplo de uso de la clase (opcional)

        >>> miclase = Clase()
        >>> miclase.cosa()
        >>> 3

        (borrame) O bien...

        .. code-block:: python

            # Comentario de python

            miclase = Clase()
            miclase.cosa()

            
    (borrame) MÁS INFORMACIÓN AQUÍ: https://docutils.sourceforge.io/docs/user/rst/quickref.html

    """


    def __init__(self):
        super(DetectionModel, self).__init__()

    def forward(self, x : torch.Tensor):
        pass