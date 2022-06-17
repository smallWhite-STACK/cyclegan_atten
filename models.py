import torch.nn as nn
import torch.nn.functional as F
import torch
from spectral import SpectralNorm

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           SELF-ATTENTION
##############################
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1),
            nn.ReLU(inplace=True)
            )

        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1),
            nn.ReLU(inplace=True)
            )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1),
            nn.ReLU(inplace=True)
            )
        #正是由于使用nn.Parameter所以gamma才能成为可以训练的参数
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma作为attention初始化的权重

        self.softmax  = nn.Softmax(dim=-1) #dim=-1指最后一个维度，即对每一个特征图的一行进行softmax
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        print(proj_query.shape)  #
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        print(proj_key.shape)    #
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        print(energy.shape)      #
        attention = self.softmax(energy) # BX (N) X (N) 
        print(attention.shape)     #
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        print(proj_value.shape)  #

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        print("最终：",out.shape)
        return out
# model = Self_Attn(64)
# x = torch.ones([1, 64, 256, 256])
# out = model(x)

##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels, out_channels, res_blocks ):
        super(GeneratorResNet, self).__init__()
        #in_channels = args.input_nc
        #out_channels = args.output_nc
        #res_blocks = args.n_residual_blocks
        # Initial convolution block
        model_0 = [   
                    nn.ReflectionPad2d(3),
                    SpectralNorm(nn.Conv2d(in_channels, 64, 7)),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        model_1 = []
        for _ in range(2):
            model_1 += [
                        SpectralNorm(nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        model_2 = []
        for _ in range(res_blocks):
            model_2 += [ResidualBlock(in_features)]

        # Upsampling
        model_3 = []
        out_features = in_features//2
        for _ in range(2):
            model_3 += [
                        SpectralNorm(nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1)),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        #在model_4加上self-atten
        self.atten_0 = Self_Attn(64)


        # Output layer
        model_4 = [  
                    nn.ReflectionPad2d(3),
                    SpectralNorm(nn.Conv2d(64, out_channels, 7)),
                    nn.Tanh() ]

        self.model_0 = nn.Sequential(*model_0)
        self.model_1 = nn.Sequential(*model_1)
        self.model_2 = nn.Sequential(*model_2)
        self.model_3 = nn.Sequential(*model_3)
        self.model_4 = nn.Sequential(*model_4)


    def forward(self, x):
        out = self.model_0(x)
        out = self.model_1(out)
        out = self.model_2(out)
        out_1 = self.model_3(out)
        print("atten前的shape",out_1.shape) #atten前的shape torch.Size([3, 256, 64, 64])

        out = self.atten_0(out_1)
        print("第一个atten后的shape",out.shape)

        out = self.model_4(out)

        return out
# model=GeneratorResNet(3,3,9)
# print(model)
# x = torch.ones([3,3,32,32])
# output = model(x)
# from torchsummary import summary
# print(summary(model,(3,256,256),device="cpu"))

##############################
#        Discriminator
##############################
class Discriminator_n_layers(nn.Module):
    def __init__(self,  n_D_layers, in_c):
        super(Discriminator_n_layers, self).__init__()

        n_layers = n_D_layers
        in_channels = in_c
        def discriminator_block(in_filters, out_filters, k=4, s=2, p=1, norm=True, sigmoid=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [SpectralNorm(nn.Conv2d(in_filters, out_filters, kernel_size=k, stride=s, padding=p))]
            if norm:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if sigmoid:
                layers.append(nn.Sigmoid())
                print('use sigmoid')
            return layers

        sequence = [*discriminator_block(in_channels, 64, norm=False)] # (1,64,128,128)

        assert n_layers<=5

        if (n_layers == 1):
            'when n_layers==1, the patch_size is (16x16)'
            out_filters = 64* 2**(n_layers-1)

        elif (1 < n_layers & n_layers<= 4):
            '''
            when n_layers==2, the patch_size is (34x34)
            when n_layers==3, the patch_size is (70x70), this is the size used in the paper
            when n_layers==4, the patch_size is (142x142)
            '''
            for k in range(1,n_layers): # k=1,2,3
                sequence += [*discriminator_block(2**(5+k), 2**(6+k))]
            out_filters = 64* 2**(n_layers-1)

        elif (n_layers == 5):
            '''
            when n_layers==5, the patch_size is (286x286), lis larger than the img_size(256),
            so this is the whole img condition
            '''
            for k in range(1,4): # k=1,2,3
                sequence += [*discriminator_block(2**(5+k), 2**(6+k))]
            # k=4
            sequence += [*discriminator_block(2**9, 2**9)] #
            out_filters = 2**9

        num_of_filter = min(2*out_filters, 2**9)

        sequence += [*discriminator_block(out_filters, num_of_filter, k=4, s=1, p=1)]
        sequence += [*discriminator_block(num_of_filter, 1, k=4, s=1, p=1, norm=False, sigmoid=False)]

        self.model = nn.Sequential(*sequence)

    def forward(self, img_input ):
        return self.model(img_input)
# model=Discriminator_n_layers(3,3)
# print(model)
# from torchsummary import summary
# print(summary(model,(3,64,64),device="cpu"))


####################################################
# Initialize generator and discriminator
####################################################
def Create_nets(args):
    generator_AB = GeneratorResNet(args.input_nc_A,   args.input_nc_B ,args.n_residual_blocks)
    discriminator_B = Discriminator_n_layers(args.n_D_layers, args.input_nc_B)
    generator_BA = GeneratorResNet(args.input_nc_B,   args.input_nc_A ,args.n_residual_blocks)
    discriminator_A = Discriminator_n_layers(args.n_D_layers, args.input_nc_A)

    if torch.cuda.is_available():
        generator_AB = generator_AB.cuda()
        discriminator_B = discriminator_B.cuda()
        generator_BA = generator_BA.cuda()
        discriminator_A = discriminator_A.cuda()


#下面是说是否使用预训练模型
    if args.epoch_start != 0:
        # Load pretrained models
        generator_AB.load_state_dict(torch.load('saved_models/%s/G__AB_%d.pth' % (opt.dataset_name, opt.epoch)))
        discriminator_B.load_state_dict(torch.load('saved_models/%s/D__B_%d.pth' % (opt.dataset_name, opt.epoch)))
        generator_BA.load_state_dict(torch.load('saved_models/%s/G__BA_%d.pth' % (opt.dataset_name, opt.epoch)))
        discriminator_A.load_state_dict(torch.load('saved_models/%s/D__A_%d.pth' % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        generator_AB.apply(weights_init_normal)
        discriminator_B.apply(weights_init_normal)
        generator_BA.apply(weights_init_normal)
        discriminator_A.apply(weights_init_normal)

    return generator_AB, discriminator_B, generator_BA, discriminator_A


from graphviz import Digraph
import torch
from torch.autograd import Variable


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


# x = torch.rand(1,3, 256, 256)
# model = GeneratorResNet(3,3,9)
# y = model(x)
# g = make_dot(y)
# g.view()

#
# model = GeneratorResNet(3,3,9)
# from torchsummary import summary
# print(summary(model,(3,256,256),device="cpu"))