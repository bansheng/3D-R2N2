# coding: utf8
def load_demo_images():
    import numpy as np
    from PIL import Image
    ims = []
    for i in range(3):
        im = Image.open('imgs/%d.png' % (i))
        # im = im.resize((127, 127))
        # im.show()
        print(type(np.array(im)[0][0])) # 127*127*3
        ims.append([np.array(im).transpose(
            (2, 0, 1)).astype(np.float32) / 255.])
    return np.array(ims)


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


from lib.config import cfg
def json_load_test(dataset_portion=[]):
    import os
    import json
    from collections import OrderedDict
    

    def model_names(model_path):
        """ Return model names"""
        model_names = [name for name in os.listdir(model_path)
                       if os.path.isdir(os.path.join(model_path, name))]
        return sorted(model_names)

    category_name_pair = []  # full path of the objs files

    # './experiments/dataset/shapenet_1000.json'  # yaml/json file that specifies a dataset (training/testing)
    cats = json.load(open(cfg.DATASET)) # 嵌套字典
    print(type(cats['04256520']))
    for cat in cats.items():
        print(cat)
    print()
    print(sorted(cats.items())[0])
    print()
    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))
    for cat in cats.items():
        print(cat)


    # for k, cat in cats.items():  # load by categories
    #     model_path = os.path.join(cfg.DIR.SHAPENET_QUERY_PATH, cat['id'])
    #     # category = cat['name']
    #     models = model_names(model_path)
    #     num_models = len(models)

    #     portioned_models = models[int(num_models * dataset_portion[0]):int(num_models *
    #                                                                        dataset_portion[1])]

    #     category_name_pair.extend([(cat['id'], model_id) for model_id in portioned_models])

    # print('lib/data_io.py: model paths from %s' % (cfg.DATASET))

    # return category_name_pair


# load_demo_images()
json_load_test(dataset_portion=cfg.TRAIN.DATASET_PORTION)
