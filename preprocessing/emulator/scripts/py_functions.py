from mako.template import Template
import numpy as  np
import skimage.transform
import os

def render_template(data,template):
    '''
    Render template
    input:
        data: dictionary with variables to render
        template: template file
    '''
    with open(template) as f:
        template = f.read()
    ##
    tmpl = Template(text=template)

    with open(os.path.join(data['output_path'], '{}.{}'.format(data['fname'],data['extension']) ),mode='w') as f:
        f.write(tmpl.render(**data))


def create_bathy(rot_z, rot_xy=0):
    width = 256
    dx = 10
    hw = int(width / 2)
    op_by_adj = np.tan(np.deg2rad(rot_z))
    height = op_by_adj * np.arange(width * 2) * dx
    z = np.tile(height, [width * 2, 1])
    rotated = skimage.transform.rotate(z, rot_xy)
    center_s = np.s_[hw:2*width-hw, hw:2*width-hw]
    z_rot = rotated[center_s]
    z_rot = z_rot - np.mean(z_rot)
    return z_rot

