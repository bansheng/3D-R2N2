# coding: utf8
def load_demo_images():
    import numpy as np
    from PIL import Image
    ims = []
    for i in range(3):
        im = Image.open('imgs/%d.jpg' % (i+3))
        im = im.resize((127, 127))
        im.show()
        ims.append([np.array(im).transpose(
            (2, 0, 1)).astype(np.float32) / 255.])
    return np.array(ims)

# load_demo_images()
# print(load_demo_images())

def scan_test1():
    import theano
    import theano.tensor as T
    import numpy

    coefficients = T.vector('coeff')
    x = T.iscalar('x')
    sum_poly_init = T.fscalar('sum_poly')
    result, update = theano.scan(lambda coefficients, power, sum_poly, x: T.cast(sum_poly +     
                                coefficients*(x**power),dtype=theano.config.floatX),
                                sequences=[coefficients, T.arange(coefficients.size)],
                                outputs_info=[sum_poly_init],
                                non_sequences=[x])

    poly_fn = theano.function([coefficients,sum_poly_init,x], result, updates=update)

    coeff_value = numpy.asarray([1.,3.,6.,5.], dtype=theano.config.floatX)
    x_value = 3
    poly_init_value = 0.
    print(poly_fn(coeff_value,poly_init_value, x_value))
    # 0+1*1 = 1
    # 1 + 3 * 3 ^ 1 = 10
    # 10 + 6 * 3 ^ 2 = 64
    # 64 + 5 * 3 ^ 3 = 199

def scan_test2():
    import theano
    import theano.tensor as T
    print('theano.scan_module.until:')
    def prod_2(pre_value, max_value):
        return pre_value*2, theano.scan_module.until(pre_value*2 > max_value)

    max_value = T.iscalar('max_value')
    result, update = theano.scan(prod_2, outputs_info=T.constant(1.),
                                non_sequences=[max_value], n_steps=100)

    prod_fn = theano.function([max_value], result, updates=update)
    print(prod_fn(400))

scan_test2()
