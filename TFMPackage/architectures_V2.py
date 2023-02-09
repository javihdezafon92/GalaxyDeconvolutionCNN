"""
Here we define the classes of the models tested in our project, based on variations of depth, width, kernel size and number of branches on the liu and lam model.
""" 

from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Add
from tensorflow.python.framework import ops
from tensorflow.keras.models import Model


class Liu_Lam_Original:
    """
    This class builds the Liu and Lam original architecture into a Keras model
    """ 
    input_shape = (64, 64, 1)
    
    def __init__(self, input_shape):
        """
        Initialize the model object's attributes

        :param tuple input_shape: shape of the input images        
        """ 
        self.input_shape = input_shape
        self.buildModel(input_shape)
        
    def getModel(self):
        """
        Get the built model
        
        :return keras.engine.functional.Functional self.model: a keras model
        """
        return self.model
    
    def buildModel(self, input_shape): 
        """
        Build the model using keras library functions

        :param tuple input_shape: shape of the input images        
        """ 
        input_img = layers.Input(shape=input_shape)

        # -----------------------------------------Upper branch (x)----------------------------------------------------
        # Compressor
        x = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)    
        x = layers.ReLU()(x)
        x1 = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(16, (4, 4), padding="same")(x1) 
        x = layers.ReLU()(x) 
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decompressor
        x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(x)
        x = layers.ReLU()(x)
        x = Add()([x1, x])
        output_up = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(x)

        # -----------------------------------------Lower branch (y)----------------------------------------------------    
        # Compressor
        y = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)
        y = layers.ReLU()(y)
        y1 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(16, (4, 4), padding="same")(y1)
        y = layers.ReLU()(y)
        y2 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(8, (4, 4), padding="same")(y2)
        y = layers.ReLU()(y)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)

        # Decompressor
        y = layers.Conv2DTranspose(16, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y2, y])
        y = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y1, y])
        output_low = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(y)

        # ---------------------------------------- End of Branches----------------------------------------------------

        pre_output = Add()([output_up, output_low])
        output = layers.ReLU()(pre_output)
        
        #output = Add()([output_up, output_low])

        # Symetric autoencoder
        self.model = Model(input_img, output)
        
        
#----------------------------------------------------------------------------------------------------------------------        
#--------------------------------------------DEPTH TESTS (Number of concolution layers)--------------------------------
#----------------------------------------------------------------------------------------------------------------------  
       
class Liu_Lam_V11:
    
    """
    This class builds a Liu and Lam architecture based model but with an aditional convolution and deconvolution layer in the lower branch
    """     
    
    input_shape = (64, 64, 1)
    
    def __init__(self, input_shape):
        """
        Initialize the model object's attributes

        :param tuple input_shape: shape of the input images        
        """         
        self.input_shape = input_shape
        self.buildModel(input_shape)
        
    def getModel(self):
        """
        Get the built model
        
        :return keras.engine.functional.Functional self.model: a keras model
        """
        return self.model
    
    def buildModel(self, input_shape):
        """
        Build the model using keras library functions

        :param tuple input_shape: shape of the input images        
        """ 
        # Input
        input_img = layers.Input(shape=input_shape)

        # -----------------------------------------Upper branch (x)----------------------------------------------------
        # Compressor
        x = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)    
        x = layers.ReLU()(x)
        x1 = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(16, (4, 4), padding="same")(x1) 
        x = layers.ReLU()(x) 
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decompressor
        x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(x)
        x = layers.ReLU()(x)
        x = Add()([x1, x])
        output_up = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(x)

        # -----------------------------------------Lower branch (y)----------------------------------------------------    
        # Compressor
        y = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)
        y = layers.ReLU()(y)
        y1 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(16, (4, 4), padding="same")(y1)
        y = layers.ReLU()(y)
        y2 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(8, (4, 4), padding="same")(y2)
        y = layers.ReLU()(y)
        y3 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(4, (4, 4), padding="same")(y3)
        y = layers.ReLU()(y)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)

        # Decompressor
        y = layers.Conv2DTranspose(8, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y3, y])
        y = layers.Conv2DTranspose(16, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y2, y])
        y = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y1, y])
        output_low = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(y)

        # ---------------------------------------- End of Branches----------------------------------------------------

        pre_output = Add()([output_up, output_low])
        output = layers.ReLU()(pre_output)

        # Symetric autoencoder
        self.model = Model(input_img, output)


class Liu_Lam_V12:
    """
    This class builds a Liu and Lam architecture based model but with an aditional convolution and deconvolution layer in both the lower branch and the upper branch
    """ 
    
    input_shape = (64, 64, 1)
    
    def __init__(self, input_shape):
        """
        Initialize the model object's attributes

        :param tuple input_shape: shape of the input images        
        """  
        self.input_shape = input_shape
        self.buildModel(input_shape)
        
    def getModel(self):
        """
        Get the built model
        
        :return keras.engine.functional.Functional self.model: a keras model
        """
        return self.model
    
    def buildModel(self, input_shape):
        """
        Build the model using keras library functions

        :param tuple input_shape: shape of the input images        
        """ 
        # Input
        input_img = layers.Input(shape=input_shape)

        # -----------------------------------------Upper branch (x)----------------------------------------------------
        # Compressor
        x = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)    
        x = layers.ReLU()(x)
        x1 = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(16, (4, 4), padding="same")(x1) 
        x = layers.ReLU()(x) 
        x2 = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(8, (4, 4), padding="same")(x2) 
        x = layers.ReLU()(x) 
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decompressor
        x = layers.Conv2DTranspose(16, (4, 4), strides=2, padding="same")(x)
        x = layers.ReLU()(x)
        x = Add()([x2, x])
        x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(x)
        x = layers.ReLU()(x)
        x = Add()([x1, x])
        output_up = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(x)

        # -----------------------------------------Lower branch (y)----------------------------------------------------    
        # Compressor
        y = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)
        y = layers.ReLU()(y)
        y1 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(16, (4, 4), padding="same")(y1)
        y = layers.ReLU()(y)
        y2 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(8, (4, 4), padding="same")(y2)
        y = layers.ReLU()(y)
        y3 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(4, (4, 4), padding="same")(y3)
        y = layers.ReLU()(y)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)

        # Decompressor
        y = layers.Conv2DTranspose(8, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y3, y])
        y = layers.Conv2DTranspose(16, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y2, y])
        y = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y1, y])
        output_low = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(y)

        # ---------------------------------------- End of Branches----------------------------------------------------

        pre_output = Add()([output_up, output_low])
        output = layers.ReLU()(pre_output)

        # Symetric autoencoder
        self.model = Model(input_img, output)
        
        
#----------------------------------------------------------------------------------------------------------------------        
#-------------------------------------------WIDTH TESTS (Number of filters)--------------------------------------------
#----------------------------------------------------------------------------------------------------------------------     

class Liu_Lam_V20:    
    """
    This class builds a Liu and Lam architecture based model but with the same number of filters (32) in each layer
    """
        
    input_shape = (64, 64, 1)
    
    def __init__(self, input_shape):
        """
        Initialize the model object's attributes

        :param tuple input_shape: shape of the input images        
        """ 
        self.input_shape = input_shape
        self.buildModel(input_shape)
        
    def getModel(self):
        """
        Get the built model
        
        :return keras.engine.functional.Functional self.model: a keras model
        """
        return self.model
    
    def buildModel(self, input_shape):
        """
        Build the model using keras library functions

        :param tuple input_shape: shape of the input images        
        """ 
        # Input
        input_img = layers.Input(shape=input_shape)

        # -----------------------------------------Upper branch (x)----------------------------------------------------
        # Compressor
        x = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)    
        x = layers.ReLU()(x)
        x1 = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(32, (4, 4), padding="same")(x1) # OJO! tienes dos RELU seguidas.
        x = layers.ReLU()(x) # ¿No debería ir X1 a la salida de la RELU? Check it.
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decompressor
        x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(x)
        x = layers.ReLU()(x)
        x = Add()([x1, x])
        output_up = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(x)

        # -----------------------------------------Lower branch (y)----------------------------------------------------    
        # Compressor
        y = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)
        y = layers.ReLU()(y)
        y1 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(32, (4, 4), padding="same")(y1)
        y = layers.ReLU()(y)
        y2 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(32, (4, 4), padding="same")(y2)
        y = layers.ReLU()(y)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)

        # Decompressor
        y = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y2, y])
        y = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y1, y])
        output_low = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(y)

        # ---------------------------------------- End of Branches----------------------------------------------------

        pre_output = Add()([output_up, output_low])
        output = layers.ReLU()(pre_output)

        # Symetric autoencoder
        self.model = Model(input_img, output)

class Liu_Lam_V21:
    """
    This class builds a Liu and Lam architecture based model but with increasing number of filters in the convolution layers
    """
    
    input_shape = (64, 64, 1)
    
    def __init__(self, input_shape):
        """
        Initialize the model object's attributes

        :param tuple input_shape: shape of the input images        
        """ 
        self.input_shape = input_shape
        self.buildModel(input_shape)
        
    def getModel(self):
        """
        Get the built model
        
        :return keras.engine.functional.Functional self.model: a keras model
        """
        return self.model
    
    def buildModel(self, input_shape):
        """
        Build the model using keras library functions

        :param tuple input_shape: shape of the input images        
        """ 
        # Input
        input_img = layers.Input(shape=input_shape)

        # -----------------------------------------Upper branch (x)----------------------------------------------------
        # Compressor
        x = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)    
        x = layers.ReLU()(x)
        x1 = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(64, (4, 4), padding="same")(x1)
        x = layers.ReLU()(x) 
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decompressor
        x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(x)
        x = layers.ReLU()(x)
        x = Add()([x1, x])
        output_up = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(x)

        # -----------------------------------------Lower branch (y)----------------------------------------------------    
        # Compressor
        y = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)
        y = layers.ReLU()(y)
        y1 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(64, (4, 4), padding="same")(y1)
        y = layers.ReLU()(y)
        y2 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(128, (4, 4), padding="same")(y2)
        y = layers.ReLU()(y)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)

        # Decompressor
        y = layers.Conv2DTranspose(64, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y2, y])
        y = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y1, y])
        output_low = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(y)

        # ---------------------------------------- End of Branches----------------------------------------------------

        pre_output = Add()([output_up, output_low])
        output = layers.ReLU()(pre_output)

        # Symetric autoencoder
        self.model = Model(input_img, output)
        
        
#----------------------------------------------------------------------------------------------------------------------        
#------------------------------------------------PRUEBAS DE TAMAÑO KERNEL----------------------------------------------
#----------------------------------------------------------------------------------------------------------------------         

class Liu_Lam_V30:
    """
    This class builds a Liu and Lam architecture based model but the kernel size is an input f_size parameter
    """
    #Código de acuerdo al paper
    input_shape = (64, 64, 1)
    f_size = 4
    
    def __init__(self, input_shape, f_size):
        """
        Initialize the model object's attributes

        :param tuple input_shape: shape of the input images
        :param int f_size: kernel size
        """
        self.input_shape = input_shape
        self.f_size = f_size
        self.buildModel(input_shape, f_size)
        
    def getModel(self):
        """
        Get the built model
        
        :return keras.engine.functional.Functional self.model: a keras model
        """
        return self.model
    
    def buildModel(self, input_shape, f_size):
        """
        Build the model using keras library functions

        :param tuple input_shape: shape of the input images
        :param int f_size: kernel size
        """ 
        # Input
        input_img = layers.Input(shape=input_shape)

        # -----------------------------------------Upper branch (x)----------------------------------------------------
        # Compressor
        x = layers.Conv2D(32, (f_size, f_size), strides=(1, 1), padding="same")(input_img)    
        x = layers.ReLU()(x)
        x1 = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(16, (f_size, f_size), padding="same")(x1) 
        x = layers.ReLU()(x) 
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decompressor
        x = layers.Conv2DTranspose(32, (f_size, f_size), strides=2, padding="same")(x)
        x = layers.ReLU()(x)
        x = Add()([x1, x])
        output_up = layers.Conv2DTranspose(1, (f_size, f_size), strides=2, activation="tanh", padding="same")(x)

        # -----------------------------------------Lower branch (y)----------------------------------------------------    
        # Compressor
        y = layers.Conv2D(32, (f_size, f_size), strides=(1, 1), padding="same")(input_img)
        y = layers.ReLU()(y)
        y1 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(16, (f_size, f_size), padding="same")(y1)
        y = layers.ReLU()(y)
        y2 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(8, (f_size, f_size), padding="same")(y2)
        y = layers.ReLU()(y)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)

        # Decompressor
        y = layers.Conv2DTranspose(16, (f_size, f_size), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y2, y])
        y = layers.Conv2DTranspose(32, (f_size, f_size), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y1, y])
        output_low = layers.Conv2DTranspose(1, (f_size, f_size), strides=2, activation="tanh", padding="same")(y)

        # ---------------------------------------- End of Branches----------------------------------------------------

        pre_output = Add()([output_up, output_low])
        output = layers.ReLU()(pre_output)

        # Symetric autoencoder
        self.model = Model(input_img, output)
        
#----------------------------------------------------------------------------------------------------------------------        
#-----------------------------------------------PRUEBAS DE MÚLTIPLES RAMAS---------------------------------------------
#----------------------------------------------------------------------------------------------------------------------   

class Liu_Lam_V41:
    """
    This class builds a Liu and Lam architecture based model but with an aditional branch of 4 convolution and deconvolution layers
    """
    
    input_shape = (64, 64, 1)
    
    def __init__(self, input_shape):
        """
        Initialize the model object's attributes

        :param tuple input_shape: shape of the input images        
        """
        self.input_shape = input_shape
        self.buildModel(input_shape)
        
    def getModel(self):
        """
        Get the built model
        
        :return keras.engine.functional.Functional self.model: a keras model
        """
        return self.model
    
    def buildModel(self, input_shape):
        """
        Build the model using keras library functions

        :param tuple input_shape: shape of the input images        
        """ 
        # Input
        input_img = layers.Input(shape=input_shape)

        # -----------------------------------------Upper branch (x)----------------------------------------------------
        # Compressor
        x = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)    
        x = layers.ReLU()(x)
        x1 = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(16, (4, 4), padding="same")(x1) 
        x = layers.ReLU()(x) 
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decompressor
        x = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(x)
        x = layers.ReLU()(x)
        x = Add()([x1, x])
        output_up = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(x)

        # -----------------------------------------Medium branch (y)----------------------------------------------------    
        # Compressor
        y = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)
        y = layers.ReLU()(y)
        y1 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(16, (4, 4), padding="same")(y1)
        y = layers.ReLU()(y)
        y2 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(8, (4, 4), padding="same")(y2)
        y = layers.ReLU()(y)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)

        # Decompressor
        y = layers.Conv2DTranspose(16, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y2, y])
        y = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y1, y])
        output_medium = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(y)

        # -----------------------------------------Lower branch (z)----------------------------------------------------    
        # Compressor
        z = layers.Conv2D(32, (4, 4), strides=(1, 1), padding="same")(input_img)
        z = layers.ReLU()(z)
        z1 = layers.MaxPooling2D((2, 2), padding="same")(z)
        z = layers.Conv2D(16, (4, 4), padding="same")(z1)
        z = layers.ReLU()(z)
        z2 = layers.MaxPooling2D((2, 2), padding="same")(z)
        z = layers.Conv2D(8, (4, 4), padding="same")(z2)
        z = layers.ReLU()(z)
        z3 = layers.MaxPooling2D((2, 2), padding="same")(z)
        z = layers.Conv2D(4, (4, 4), padding="same")(z3)
        z = layers.ReLU()(z)
        z = layers.MaxPooling2D((2, 2), padding="same")(z)

        # Decompressor
        z = layers.Conv2DTranspose(8, (4, 4), strides=2, padding="same")(z)
        z = layers.ReLU()(z)
        z = Add()([z3, z])
        z = layers.Conv2DTranspose(16, (4, 4), strides=2, padding="same")(z)
        z = layers.ReLU()(z)
        z = Add()([z2, z])
        z = layers.Conv2DTranspose(32, (4, 4), strides=2, padding="same")(z)
        z = layers.ReLU()(z)
        z = Add()([z1, z])
        output_low = layers.Conv2DTranspose(1, (4, 4), strides=2, activation="tanh", padding="same")(z)

        # ---------------------------------------- End of Branches----------------------------------------------------

        pre_output = Add()([output_up, output_medium, output_low])
        output = layers.ReLU()(pre_output)

        # Symetric autoencoder
        self.model = Model(input_img, output)
        
#----------------------------------------------------------------------------------------------------------------------        
#---------------------------------------------------ARQUITECTURA FINAL-------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------           

class My_model:
    """
    This class builds our final model, which consists of the Liu and Lam architecture whit an aditional convolution and deconvolution layer in the lower branch, increasing number of filters and kernel size of 5
    """
    
    
    input_shape = (64, 64, 1)
    
    def __init__(self, input_shape):
        """
        Initialize the model object's attributes

        :param tuple input_shape: shape of the input images        
        """
        self.input_shape = input_shape
        self.buildModel(input_shape)
        
    def getModel(self):
        """
        Get the built model
        
        :return keras.engine.functional.Functional self.model: a keras model
        """
        return self.model
    
    def buildModel(self, input_shape):
        """
        Build the model using keras library functions

        :param tuple input_shape: shape of the input images        
        """ 
        # Input
        input_img = layers.Input(shape=input_shape)

        # -----------------------------------------Upper branch (x)----------------------------------------------------
        # Compressor
        x = layers.Conv2D(32, (5, 5), strides=(1, 1), padding="same")(input_img)    
        x = layers.ReLU()(x)
        x1 = layers.MaxPooling2D((2, 2), padding="same")(x)
        x = layers.Conv2D(64, (5, 5), padding="same")(x1) 
        x = layers.ReLU()(x) 
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        # Decompressor
        x = layers.Conv2DTranspose(32, (5, 5), strides=2, padding="same")(x)
        x = layers.ReLU()(x)
        x = Add()([x1, x])
        output_up = layers.Conv2DTranspose(1, (5, 5), strides=2, activation="tanh", padding="same")(x)

        # -----------------------------------------Lower branch (y)----------------------------------------------------    
        # Compressor
        y = layers.Conv2D(32, (5, 5), strides=(1, 1), padding="same")(input_img)
        y = layers.ReLU()(y)
        y1 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(64, (5, 5), padding="same")(y1)
        y = layers.ReLU()(y)
        y2 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(128, (5, 5), padding="same")(y2)
        y = layers.ReLU()(y)
        y3 = layers.MaxPooling2D((2, 2), padding="same")(y)
        y = layers.Conv2D(256, (5, 5), padding="same")(y3)
        y = layers.ReLU()(y)
        y = layers.MaxPooling2D((2, 2), padding="same")(y)

        # Decompressor
        y = layers.Conv2DTranspose(128, (5, 5), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y3, y])
        y = layers.Conv2DTranspose(64, (5, 5), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y2, y])
        y = layers.Conv2DTranspose(32, (5, 5), strides=2, padding="same")(y)
        y = layers.ReLU()(y)
        y = Add()([y1, y])
        output_low = layers.Conv2DTranspose(1, (5, 5), strides=2, activation="tanh", padding="same")(y)

        # ---------------------------------------- End of Branches----------------------------------------------------

        pre_output = Add()([output_up, output_low])
        output = layers.ReLU()(pre_output)
        
        #output = Add()([output_up, output_low])

        # Symetric autoencoder
        self.model = Model(input_img, output)