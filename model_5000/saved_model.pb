��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
;
Elu
features"T
activations"T"
Ttype:
2
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
u
Input/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
�*
shared_nameInput/kernel
n
 Input/kernel/Read/ReadVariableOpReadVariableOpInput/kernel*
_output_shapes
:	
�*
dtype0
m

Input/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
Input/bias
f
Input/bias/Read/ReadVariableOpReadVariableOp
Input/bias*
_output_shapes	
:�*
dtype0
y
Hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_nameHidden1/kernel
r
"Hidden1/kernel/Read/ReadVariableOpReadVariableOpHidden1/kernel*
_output_shapes
:	�@*
dtype0
p
Hidden1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameHidden1/bias
i
 Hidden1/bias/Read/ReadVariableOpReadVariableOpHidden1/bias*
_output_shapes
:@*
dtype0
x
Hidden2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_nameHidden2/kernel
q
"Hidden2/kernel/Read/ReadVariableOpReadVariableOpHidden2/kernel*
_output_shapes

:@*
dtype0
p
Hidden2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameHidden2/bias
i
 Hidden2/bias/Read/ReadVariableOpReadVariableOpHidden2/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
 
�
!layer_regularization_losses
"metrics
	variables
#layer_metrics

$layers
%non_trainable_variables
trainable_variables
regularization_losses
 
XV
VARIABLE_VALUEInput/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
Input/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
&layer_regularization_losses
'metrics
	variables
(layer_metrics

)layers
*non_trainable_variables
trainable_variables
regularization_losses
 
 
 
�
+layer_regularization_losses
,metrics
	variables
-layer_metrics

.layers
/non_trainable_variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEHidden1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEHidden1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
0layer_regularization_losses
1metrics
	variables
2layer_metrics

3layers
4non_trainable_variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEHidden2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEHidden2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
5layer_regularization_losses
6metrics
	variables
7layer_metrics

8layers
9non_trainable_variables
trainable_variables
regularization_losses
 
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_Input_inputPlaceholder*+
_output_shapes
:���������
*
dtype0* 
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_Input_inputInput/kernel
Input/biasHidden1/kernelHidden1/biasHidden2/kernelHidden2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference_signature_wrapper_904
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename Input/kernel/Read/ReadVariableOpInput/bias/Read/ReadVariableOp"Hidden1/kernel/Read/ReadVariableOp Hidden1/bias/Read/ReadVariableOp"Hidden2/kernel/Read/ReadVariableOp Hidden2/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_1143
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameInput/kernel
Input/biasHidden1/kernelHidden1/biasHidden2/kernelHidden2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_1171��
�
�
)__inference_sequential_layer_call_fn_1007

inputs
input_kernel

input_bias
hidden1_kernel
hidden1_bias
hidden2_kernel
hidden2_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinput_kernel
input_biashidden1_kernelhidden1_biashidden2_kernelhidden2_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_8572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
A__inference_Hidden2_layer_call_and_return_conditional_losses_1095

inputs(
$matmul_readvariableop_hidden2_kernel'
#biasadd_readvariableop_hidden2_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_hidden2_kernel*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_hidden2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_sequential_layer_call_fn_891
input_input
input_kernel

input_bias
hidden1_kernel
hidden1_bias
hidden2_kernel
hidden2_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_inputinput_kernel
input_biashidden1_kernelhidden1_biashidden2_kernelhidden2_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_8822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������

%
_user_specified_nameInput_input
�
]
A__inference_Flatten1_layer_call_and_return_conditional_losses_772

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
&__inference_Hidden1_layer_call_fn_1085

inputs
hidden1_kernel
hidden1_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputshidden1_kernelhidden1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_Hidden1_layer_call_and_return_conditional_losses_7912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
 __inference__traced_restore_1171
file_prefix!
assignvariableop_input_kernel!
assignvariableop_1_input_bias%
!assignvariableop_2_hidden1_kernel#
assignvariableop_3_hidden1_bias%
!assignvariableop_4_hidden2_kernel#
assignvariableop_5_hidden2_bias

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_input_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_input_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_hidden1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_hidden1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_hidden2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_hidden2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6�

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�D
�
__inference__wrapped_model_719
input_input:
6sequential_input_tensordot_readvariableop_input_kernel6
2sequential_input_biasadd_readvariableop_input_bias;
7sequential_hidden1_matmul_readvariableop_hidden1_kernel:
6sequential_hidden1_biasadd_readvariableop_hidden1_bias;
7sequential_hidden2_matmul_readvariableop_hidden2_kernel:
6sequential_hidden2_biasadd_readvariableop_hidden2_bias
identity��)sequential/Hidden1/BiasAdd/ReadVariableOp�(sequential/Hidden1/MatMul/ReadVariableOp�)sequential/Hidden2/BiasAdd/ReadVariableOp�(sequential/Hidden2/MatMul/ReadVariableOp�'sequential/Input/BiasAdd/ReadVariableOp�)sequential/Input/Tensordot/ReadVariableOp�
)sequential/Input/Tensordot/ReadVariableOpReadVariableOp6sequential_input_tensordot_readvariableop_input_kernel*
_output_shapes
:	
�*
dtype02+
)sequential/Input/Tensordot/ReadVariableOp�
sequential/Input/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
sequential/Input/Tensordot/axes�
sequential/Input/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2!
sequential/Input/Tensordot/free
 sequential/Input/Tensordot/ShapeShapeinput_input*
T0*
_output_shapes
:2"
 sequential/Input/Tensordot/Shape�
(sequential/Input/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/Input/Tensordot/GatherV2/axis�
#sequential/Input/Tensordot/GatherV2GatherV2)sequential/Input/Tensordot/Shape:output:0(sequential/Input/Tensordot/free:output:01sequential/Input/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#sequential/Input/Tensordot/GatherV2�
*sequential/Input/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/Input/Tensordot/GatherV2_1/axis�
%sequential/Input/Tensordot/GatherV2_1GatherV2)sequential/Input/Tensordot/Shape:output:0(sequential/Input/Tensordot/axes:output:03sequential/Input/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/Input/Tensordot/GatherV2_1�
 sequential/Input/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 sequential/Input/Tensordot/Const�
sequential/Input/Tensordot/ProdProd,sequential/Input/Tensordot/GatherV2:output:0)sequential/Input/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
sequential/Input/Tensordot/Prod�
"sequential/Input/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/Input/Tensordot/Const_1�
!sequential/Input/Tensordot/Prod_1Prod.sequential/Input/Tensordot/GatherV2_1:output:0+sequential/Input/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!sequential/Input/Tensordot/Prod_1�
&sequential/Input/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/Input/Tensordot/concat/axis�
!sequential/Input/Tensordot/concatConcatV2(sequential/Input/Tensordot/free:output:0(sequential/Input/Tensordot/axes:output:0/sequential/Input/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!sequential/Input/Tensordot/concat�
 sequential/Input/Tensordot/stackPack(sequential/Input/Tensordot/Prod:output:0*sequential/Input/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 sequential/Input/Tensordot/stack�
$sequential/Input/Tensordot/transpose	Transposeinput_input*sequential/Input/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������
2&
$sequential/Input/Tensordot/transpose�
"sequential/Input/Tensordot/ReshapeReshape(sequential/Input/Tensordot/transpose:y:0)sequential/Input/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2$
"sequential/Input/Tensordot/Reshape�
!sequential/Input/Tensordot/MatMulMatMul+sequential/Input/Tensordot/Reshape:output:01sequential/Input/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2#
!sequential/Input/Tensordot/MatMul�
"sequential/Input/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2$
"sequential/Input/Tensordot/Const_2�
(sequential/Input/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/Input/Tensordot/concat_1/axis�
#sequential/Input/Tensordot/concat_1ConcatV2,sequential/Input/Tensordot/GatherV2:output:0+sequential/Input/Tensordot/Const_2:output:01sequential/Input/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/Input/Tensordot/concat_1�
sequential/Input/TensordotReshape+sequential/Input/Tensordot/MatMul:product:0,sequential/Input/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
sequential/Input/Tensordot�
'sequential/Input/BiasAdd/ReadVariableOpReadVariableOp2sequential_input_biasadd_readvariableop_input_bias*
_output_shapes	
:�*
dtype02)
'sequential/Input/BiasAdd/ReadVariableOp�
sequential/Input/BiasAddBiasAdd#sequential/Input/Tensordot:output:0/sequential/Input/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
sequential/Input/BiasAdd�
sequential/Input/EluElu!sequential/Input/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
sequential/Input/Elu�
sequential/Flatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
sequential/Flatten1/Const�
sequential/Flatten1/ReshapeReshape"sequential/Input/Elu:activations:0"sequential/Flatten1/Const:output:0*
T0*(
_output_shapes
:����������2
sequential/Flatten1/Reshape�
(sequential/Hidden1/MatMul/ReadVariableOpReadVariableOp7sequential_hidden1_matmul_readvariableop_hidden1_kernel*
_output_shapes
:	�@*
dtype02*
(sequential/Hidden1/MatMul/ReadVariableOp�
sequential/Hidden1/MatMulMatMul$sequential/Flatten1/Reshape:output:00sequential/Hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
sequential/Hidden1/MatMul�
)sequential/Hidden1/BiasAdd/ReadVariableOpReadVariableOp6sequential_hidden1_biasadd_readvariableop_hidden1_bias*
_output_shapes
:@*
dtype02+
)sequential/Hidden1/BiasAdd/ReadVariableOp�
sequential/Hidden1/BiasAddBiasAdd#sequential/Hidden1/MatMul:product:01sequential/Hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
sequential/Hidden1/BiasAdd�
sequential/Hidden1/EluElu#sequential/Hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
sequential/Hidden1/Elu�
(sequential/Hidden2/MatMul/ReadVariableOpReadVariableOp7sequential_hidden2_matmul_readvariableop_hidden2_kernel*
_output_shapes

:@*
dtype02*
(sequential/Hidden2/MatMul/ReadVariableOp�
sequential/Hidden2/MatMulMatMul$sequential/Hidden1/Elu:activations:00sequential/Hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/Hidden2/MatMul�
)sequential/Hidden2/BiasAdd/ReadVariableOpReadVariableOp6sequential_hidden2_biasadd_readvariableop_hidden2_bias*
_output_shapes
:*
dtype02+
)sequential/Hidden2/BiasAdd/ReadVariableOp�
sequential/Hidden2/BiasAddBiasAdd#sequential/Hidden2/MatMul:product:01sequential/Hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/Hidden2/BiasAdd�
IdentityIdentity#sequential/Hidden2/BiasAdd:output:0*^sequential/Hidden1/BiasAdd/ReadVariableOp)^sequential/Hidden1/MatMul/ReadVariableOp*^sequential/Hidden2/BiasAdd/ReadVariableOp)^sequential/Hidden2/MatMul/ReadVariableOp(^sequential/Input/BiasAdd/ReadVariableOp*^sequential/Input/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::2V
)sequential/Hidden1/BiasAdd/ReadVariableOp)sequential/Hidden1/BiasAdd/ReadVariableOp2T
(sequential/Hidden1/MatMul/ReadVariableOp(sequential/Hidden1/MatMul/ReadVariableOp2V
)sequential/Hidden2/BiasAdd/ReadVariableOp)sequential/Hidden2/BiasAdd/ReadVariableOp2T
(sequential/Hidden2/MatMul/ReadVariableOp(sequential/Hidden2/MatMul/ReadVariableOp2R
'sequential/Input/BiasAdd/ReadVariableOp'sequential/Input/BiasAdd/ReadVariableOp2V
)sequential/Input/Tensordot/ReadVariableOp)sequential/Input/Tensordot/ReadVariableOp:X T
+
_output_shapes
:���������

%
_user_specified_nameInput_input
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_840
input_input
input_input_kernel
input_input_bias
hidden1_hidden1_kernel
hidden1_hidden1_bias
hidden2_hidden2_kernel
hidden2_hidden2_bias
identity��Hidden1/StatefulPartitionedCall�Hidden2/StatefulPartitionedCall�Input/StatefulPartitionedCall�
Input/StatefulPartitionedCallStatefulPartitionedCallinput_inputinput_input_kernelinput_input_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_Input_layer_call_and_return_conditional_losses_7542
Input/StatefulPartitionedCall�
Flatten1/PartitionedCallPartitionedCall&Input/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Flatten1_layer_call_and_return_conditional_losses_7722
Flatten1/PartitionedCall�
Hidden1/StatefulPartitionedCallStatefulPartitionedCall!Flatten1/PartitionedCall:output:0hidden1_hidden1_kernelhidden1_hidden1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_Hidden1_layer_call_and_return_conditional_losses_7912!
Hidden1/StatefulPartitionedCall�
Hidden2/StatefulPartitionedCallStatefulPartitionedCall(Hidden1/StatefulPartitionedCall:output:0hidden2_hidden2_kernelhidden2_hidden2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_Hidden2_layer_call_and_return_conditional_losses_8132!
Hidden2/StatefulPartitionedCall�
IdentityIdentity(Hidden2/StatefulPartitionedCall:output:0 ^Hidden1/StatefulPartitionedCall ^Hidden2/StatefulPartitionedCall^Input/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::2B
Hidden1/StatefulPartitionedCallHidden1/StatefulPartitionedCall2B
Hidden2/StatefulPartitionedCallHidden2/StatefulPartitionedCall2>
Input/StatefulPartitionedCallInput/StatefulPartitionedCall:X T
+
_output_shapes
:���������

%
_user_specified_nameInput_input
�
�
__inference__traced_save_1143
file_prefix+
'savev2_input_kernel_read_readvariableop)
%savev2_input_bias_read_readvariableop-
)savev2_hidden1_kernel_read_readvariableop+
'savev2_hidden1_bias_read_readvariableop-
)savev2_hidden2_kernel_read_readvariableop+
'savev2_hidden2_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_input_kernel_read_readvariableop%savev2_input_bias_read_readvariableop)savev2_hidden1_kernel_read_readvariableop'savev2_hidden1_bias_read_readvariableop)savev2_hidden2_kernel_read_readvariableop'savev2_hidden2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*J
_input_shapes9
7: :	
�:�:	�@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	
�:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
�
^
B__inference_Flatten1_layer_call_and_return_conditional_losses_1062

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_857

inputs
input_input_kernel
input_input_bias
hidden1_hidden1_kernel
hidden1_hidden1_bias
hidden2_hidden2_kernel
hidden2_hidden2_bias
identity��Hidden1/StatefulPartitionedCall�Hidden2/StatefulPartitionedCall�Input/StatefulPartitionedCall�
Input/StatefulPartitionedCallStatefulPartitionedCallinputsinput_input_kernelinput_input_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_Input_layer_call_and_return_conditional_losses_7542
Input/StatefulPartitionedCall�
Flatten1/PartitionedCallPartitionedCall&Input/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Flatten1_layer_call_and_return_conditional_losses_7722
Flatten1/PartitionedCall�
Hidden1/StatefulPartitionedCallStatefulPartitionedCall!Flatten1/PartitionedCall:output:0hidden1_hidden1_kernelhidden1_hidden1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_Hidden1_layer_call_and_return_conditional_losses_7912!
Hidden1/StatefulPartitionedCall�
Hidden2/StatefulPartitionedCallStatefulPartitionedCall(Hidden1/StatefulPartitionedCall:output:0hidden2_hidden2_kernelhidden2_hidden2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_Hidden2_layer_call_and_return_conditional_losses_8132!
Hidden2/StatefulPartitionedCall�
IdentityIdentity(Hidden2/StatefulPartitionedCall:output:0 ^Hidden1/StatefulPartitionedCall ^Hidden2/StatefulPartitionedCall^Input/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::2B
Hidden1/StatefulPartitionedCallHidden1/StatefulPartitionedCall2B
Hidden2/StatefulPartitionedCallHidden2/StatefulPartitionedCall2>
Input/StatefulPartitionedCallInput/StatefulPartitionedCall:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�7
�
C__inference_sequential_layer_call_and_return_conditional_losses_950

inputs/
+input_tensordot_readvariableop_input_kernel+
'input_biasadd_readvariableop_input_bias0
,hidden1_matmul_readvariableop_hidden1_kernel/
+hidden1_biasadd_readvariableop_hidden1_bias0
,hidden2_matmul_readvariableop_hidden2_kernel/
+hidden2_biasadd_readvariableop_hidden2_bias
identity��Hidden1/BiasAdd/ReadVariableOp�Hidden1/MatMul/ReadVariableOp�Hidden2/BiasAdd/ReadVariableOp�Hidden2/MatMul/ReadVariableOp�Input/BiasAdd/ReadVariableOp�Input/Tensordot/ReadVariableOp�
Input/Tensordot/ReadVariableOpReadVariableOp+input_tensordot_readvariableop_input_kernel*
_output_shapes
:	
�*
dtype02 
Input/Tensordot/ReadVariableOpv
Input/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Input/Tensordot/axes}
Input/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Input/Tensordot/freed
Input/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Input/Tensordot/Shape�
Input/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Input/Tensordot/GatherV2/axis�
Input/Tensordot/GatherV2GatherV2Input/Tensordot/Shape:output:0Input/Tensordot/free:output:0&Input/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Input/Tensordot/GatherV2�
Input/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
Input/Tensordot/GatherV2_1/axis�
Input/Tensordot/GatherV2_1GatherV2Input/Tensordot/Shape:output:0Input/Tensordot/axes:output:0(Input/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Input/Tensordot/GatherV2_1x
Input/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Input/Tensordot/Const�
Input/Tensordot/ProdProd!Input/Tensordot/GatherV2:output:0Input/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Input/Tensordot/Prod|
Input/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Input/Tensordot/Const_1�
Input/Tensordot/Prod_1Prod#Input/Tensordot/GatherV2_1:output:0 Input/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Input/Tensordot/Prod_1|
Input/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Input/Tensordot/concat/axis�
Input/Tensordot/concatConcatV2Input/Tensordot/free:output:0Input/Tensordot/axes:output:0$Input/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Input/Tensordot/concat�
Input/Tensordot/stackPackInput/Tensordot/Prod:output:0Input/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Input/Tensordot/stack�
Input/Tensordot/transpose	TransposeinputsInput/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������
2
Input/Tensordot/transpose�
Input/Tensordot/ReshapeReshapeInput/Tensordot/transpose:y:0Input/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Input/Tensordot/Reshape�
Input/Tensordot/MatMulMatMul Input/Tensordot/Reshape:output:0&Input/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Input/Tensordot/MatMul}
Input/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Input/Tensordot/Const_2�
Input/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Input/Tensordot/concat_1/axis�
Input/Tensordot/concat_1ConcatV2!Input/Tensordot/GatherV2:output:0 Input/Tensordot/Const_2:output:0&Input/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Input/Tensordot/concat_1�
Input/TensordotReshape Input/Tensordot/MatMul:product:0!Input/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
Input/Tensordot�
Input/BiasAdd/ReadVariableOpReadVariableOp'input_biasadd_readvariableop_input_bias*
_output_shapes	
:�*
dtype02
Input/BiasAdd/ReadVariableOp�
Input/BiasAddBiasAddInput/Tensordot:output:0$Input/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
Input/BiasAddl
	Input/EluEluInput/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
	Input/Eluq
Flatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Flatten1/Const�
Flatten1/ReshapeReshapeInput/Elu:activations:0Flatten1/Const:output:0*
T0*(
_output_shapes
:����������2
Flatten1/Reshape�
Hidden1/MatMul/ReadVariableOpReadVariableOp,hidden1_matmul_readvariableop_hidden1_kernel*
_output_shapes
:	�@*
dtype02
Hidden1/MatMul/ReadVariableOp�
Hidden1/MatMulMatMulFlatten1/Reshape:output:0%Hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Hidden1/MatMul�
Hidden1/BiasAdd/ReadVariableOpReadVariableOp+hidden1_biasadd_readvariableop_hidden1_bias*
_output_shapes
:@*
dtype02 
Hidden1/BiasAdd/ReadVariableOp�
Hidden1/BiasAddBiasAddHidden1/MatMul:product:0&Hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Hidden1/BiasAddm
Hidden1/EluEluHidden1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Hidden1/Elu�
Hidden2/MatMul/ReadVariableOpReadVariableOp,hidden2_matmul_readvariableop_hidden2_kernel*
_output_shapes

:@*
dtype02
Hidden2/MatMul/ReadVariableOp�
Hidden2/MatMulMatMulHidden1/Elu:activations:0%Hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Hidden2/MatMul�
Hidden2/BiasAdd/ReadVariableOpReadVariableOp+hidden2_biasadd_readvariableop_hidden2_bias*
_output_shapes
:*
dtype02 
Hidden2/BiasAdd/ReadVariableOp�
Hidden2/BiasAddBiasAddHidden2/MatMul:product:0&Hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Hidden2/BiasAdd�
IdentityIdentityHidden2/BiasAdd:output:0^Hidden1/BiasAdd/ReadVariableOp^Hidden1/MatMul/ReadVariableOp^Hidden2/BiasAdd/ReadVariableOp^Hidden2/MatMul/ReadVariableOp^Input/BiasAdd/ReadVariableOp^Input/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::2@
Hidden1/BiasAdd/ReadVariableOpHidden1/BiasAdd/ReadVariableOp2>
Hidden1/MatMul/ReadVariableOpHidden1/MatMul/ReadVariableOp2@
Hidden2/BiasAdd/ReadVariableOpHidden2/BiasAdd/ReadVariableOp2>
Hidden2/MatMul/ReadVariableOpHidden2/MatMul/ReadVariableOp2<
Input/BiasAdd/ReadVariableOpInput/BiasAdd/ReadVariableOp2@
Input/Tensordot/ReadVariableOpInput/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
!__inference_signature_wrapper_904
input_input
input_kernel

input_bias
hidden1_kernel
hidden1_bias
hidden2_kernel
hidden2_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_inputinput_kernel
input_biashidden1_kernelhidden1_biashidden2_kernelhidden2_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__wrapped_model_7192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������

%
_user_specified_nameInput_input
�
C
'__inference_Flatten1_layer_call_fn_1067

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Flatten1_layer_call_and_return_conditional_losses_7722
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_882

inputs
input_input_kernel
input_input_bias
hidden1_hidden1_kernel
hidden1_hidden1_bias
hidden2_hidden2_kernel
hidden2_hidden2_bias
identity��Hidden1/StatefulPartitionedCall�Hidden2/StatefulPartitionedCall�Input/StatefulPartitionedCall�
Input/StatefulPartitionedCallStatefulPartitionedCallinputsinput_input_kernelinput_input_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_Input_layer_call_and_return_conditional_losses_7542
Input/StatefulPartitionedCall�
Flatten1/PartitionedCallPartitionedCall&Input/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Flatten1_layer_call_and_return_conditional_losses_7722
Flatten1/PartitionedCall�
Hidden1/StatefulPartitionedCallStatefulPartitionedCall!Flatten1/PartitionedCall:output:0hidden1_hidden1_kernelhidden1_hidden1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_Hidden1_layer_call_and_return_conditional_losses_7912!
Hidden1/StatefulPartitionedCall�
Hidden2/StatefulPartitionedCallStatefulPartitionedCall(Hidden1/StatefulPartitionedCall:output:0hidden2_hidden2_kernelhidden2_hidden2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_Hidden2_layer_call_and_return_conditional_losses_8132!
Hidden2/StatefulPartitionedCall�
IdentityIdentity(Hidden2/StatefulPartitionedCall:output:0 ^Hidden1/StatefulPartitionedCall ^Hidden2/StatefulPartitionedCall^Input/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::2B
Hidden1/StatefulPartitionedCallHidden1/StatefulPartitionedCall2B
Hidden2/StatefulPartitionedCallHidden2/StatefulPartitionedCall2>
Input/StatefulPartitionedCallInput/StatefulPartitionedCall:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
&__inference_Hidden2_layer_call_fn_1102

inputs
hidden2_kernel
hidden2_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputshidden2_kernelhidden2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_Hidden2_layer_call_and_return_conditional_losses_8132
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�7
�
C__inference_sequential_layer_call_and_return_conditional_losses_996

inputs/
+input_tensordot_readvariableop_input_kernel+
'input_biasadd_readvariableop_input_bias0
,hidden1_matmul_readvariableop_hidden1_kernel/
+hidden1_biasadd_readvariableop_hidden1_bias0
,hidden2_matmul_readvariableop_hidden2_kernel/
+hidden2_biasadd_readvariableop_hidden2_bias
identity��Hidden1/BiasAdd/ReadVariableOp�Hidden1/MatMul/ReadVariableOp�Hidden2/BiasAdd/ReadVariableOp�Hidden2/MatMul/ReadVariableOp�Input/BiasAdd/ReadVariableOp�Input/Tensordot/ReadVariableOp�
Input/Tensordot/ReadVariableOpReadVariableOp+input_tensordot_readvariableop_input_kernel*
_output_shapes
:	
�*
dtype02 
Input/Tensordot/ReadVariableOpv
Input/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Input/Tensordot/axes}
Input/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Input/Tensordot/freed
Input/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Input/Tensordot/Shape�
Input/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Input/Tensordot/GatherV2/axis�
Input/Tensordot/GatherV2GatherV2Input/Tensordot/Shape:output:0Input/Tensordot/free:output:0&Input/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Input/Tensordot/GatherV2�
Input/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
Input/Tensordot/GatherV2_1/axis�
Input/Tensordot/GatherV2_1GatherV2Input/Tensordot/Shape:output:0Input/Tensordot/axes:output:0(Input/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Input/Tensordot/GatherV2_1x
Input/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Input/Tensordot/Const�
Input/Tensordot/ProdProd!Input/Tensordot/GatherV2:output:0Input/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Input/Tensordot/Prod|
Input/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Input/Tensordot/Const_1�
Input/Tensordot/Prod_1Prod#Input/Tensordot/GatherV2_1:output:0 Input/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Input/Tensordot/Prod_1|
Input/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Input/Tensordot/concat/axis�
Input/Tensordot/concatConcatV2Input/Tensordot/free:output:0Input/Tensordot/axes:output:0$Input/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Input/Tensordot/concat�
Input/Tensordot/stackPackInput/Tensordot/Prod:output:0Input/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Input/Tensordot/stack�
Input/Tensordot/transpose	TransposeinputsInput/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������
2
Input/Tensordot/transpose�
Input/Tensordot/ReshapeReshapeInput/Tensordot/transpose:y:0Input/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Input/Tensordot/Reshape�
Input/Tensordot/MatMulMatMul Input/Tensordot/Reshape:output:0&Input/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Input/Tensordot/MatMul}
Input/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Input/Tensordot/Const_2�
Input/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Input/Tensordot/concat_1/axis�
Input/Tensordot/concat_1ConcatV2!Input/Tensordot/GatherV2:output:0 Input/Tensordot/Const_2:output:0&Input/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Input/Tensordot/concat_1�
Input/TensordotReshape Input/Tensordot/MatMul:product:0!Input/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
Input/Tensordot�
Input/BiasAdd/ReadVariableOpReadVariableOp'input_biasadd_readvariableop_input_bias*
_output_shapes	
:�*
dtype02
Input/BiasAdd/ReadVariableOp�
Input/BiasAddBiasAddInput/Tensordot:output:0$Input/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2
Input/BiasAddl
	Input/EluEluInput/BiasAdd:output:0*
T0*,
_output_shapes
:����������2
	Input/Eluq
Flatten1/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Flatten1/Const�
Flatten1/ReshapeReshapeInput/Elu:activations:0Flatten1/Const:output:0*
T0*(
_output_shapes
:����������2
Flatten1/Reshape�
Hidden1/MatMul/ReadVariableOpReadVariableOp,hidden1_matmul_readvariableop_hidden1_kernel*
_output_shapes
:	�@*
dtype02
Hidden1/MatMul/ReadVariableOp�
Hidden1/MatMulMatMulFlatten1/Reshape:output:0%Hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Hidden1/MatMul�
Hidden1/BiasAdd/ReadVariableOpReadVariableOp+hidden1_biasadd_readvariableop_hidden1_bias*
_output_shapes
:@*
dtype02 
Hidden1/BiasAdd/ReadVariableOp�
Hidden1/BiasAddBiasAddHidden1/MatMul:product:0&Hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Hidden1/BiasAddm
Hidden1/EluEluHidden1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Hidden1/Elu�
Hidden2/MatMul/ReadVariableOpReadVariableOp,hidden2_matmul_readvariableop_hidden2_kernel*
_output_shapes

:@*
dtype02
Hidden2/MatMul/ReadVariableOp�
Hidden2/MatMulMatMulHidden1/Elu:activations:0%Hidden2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Hidden2/MatMul�
Hidden2/BiasAdd/ReadVariableOpReadVariableOp+hidden2_biasadd_readvariableop_hidden2_bias*
_output_shapes
:*
dtype02 
Hidden2/BiasAdd/ReadVariableOp�
Hidden2/BiasAddBiasAddHidden2/MatMul:product:0&Hidden2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Hidden2/BiasAdd�
IdentityIdentityHidden2/BiasAdd:output:0^Hidden1/BiasAdd/ReadVariableOp^Hidden1/MatMul/ReadVariableOp^Hidden2/BiasAdd/ReadVariableOp^Hidden2/MatMul/ReadVariableOp^Input/BiasAdd/ReadVariableOp^Input/Tensordot/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::2@
Hidden1/BiasAdd/ReadVariableOpHidden1/BiasAdd/ReadVariableOp2>
Hidden1/MatMul/ReadVariableOpHidden1/MatMul/ReadVariableOp2@
Hidden2/BiasAdd/ReadVariableOpHidden2/BiasAdd/ReadVariableOp2>
Hidden2/MatMul/ReadVariableOpHidden2/MatMul/ReadVariableOp2<
Input/BiasAdd/ReadVariableOpInput/BiasAdd/ReadVariableOp2@
Input/Tensordot/ReadVariableOpInput/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
C__inference_sequential_layer_call_and_return_conditional_losses_826
input_input
input_input_kernel
input_input_bias
hidden1_hidden1_kernel
hidden1_hidden1_bias
hidden2_hidden2_kernel
hidden2_hidden2_bias
identity��Hidden1/StatefulPartitionedCall�Hidden2/StatefulPartitionedCall�Input/StatefulPartitionedCall�
Input/StatefulPartitionedCallStatefulPartitionedCallinput_inputinput_input_kernelinput_input_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_Input_layer_call_and_return_conditional_losses_7542
Input/StatefulPartitionedCall�
Flatten1/PartitionedCallPartitionedCall&Input/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_Flatten1_layer_call_and_return_conditional_losses_7722
Flatten1/PartitionedCall�
Hidden1/StatefulPartitionedCallStatefulPartitionedCall!Flatten1/PartitionedCall:output:0hidden1_hidden1_kernelhidden1_hidden1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_Hidden1_layer_call_and_return_conditional_losses_7912!
Hidden1/StatefulPartitionedCall�
Hidden2/StatefulPartitionedCallStatefulPartitionedCall(Hidden1/StatefulPartitionedCall:output:0hidden2_hidden2_kernelhidden2_hidden2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_Hidden2_layer_call_and_return_conditional_losses_8132!
Hidden2/StatefulPartitionedCall�
IdentityIdentity(Hidden2/StatefulPartitionedCall:output:0 ^Hidden1/StatefulPartitionedCall ^Hidden2/StatefulPartitionedCall^Input/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::2B
Hidden1/StatefulPartitionedCallHidden1/StatefulPartitionedCall2B
Hidden2/StatefulPartitionedCallHidden2/StatefulPartitionedCall2>
Input/StatefulPartitionedCallInput/StatefulPartitionedCall:X T
+
_output_shapes
:���������

%
_user_specified_nameInput_input
�
�
)__inference_sequential_layer_call_fn_1018

inputs
input_kernel

input_bias
hidden1_kernel
hidden1_bias
hidden2_kernel
hidden2_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinput_kernel
input_biashidden1_kernelhidden1_biashidden2_kernelhidden2_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_8822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
� 
�
?__inference_Input_layer_call_and_return_conditional_losses_1049

inputs)
%tensordot_readvariableop_input_kernel%
!biasadd_readvariableop_input_bias
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp%tensordot_readvariableop_input_kernel*
_output_shapes
:	
�*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������
2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_input_bias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAddZ
EluEluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Elu�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
(__inference_sequential_layer_call_fn_866
input_input
input_kernel

input_bias
hidden1_kernel
hidden1_bias
hidden2_kernel
hidden2_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_inputinput_kernel
input_biashidden1_kernelhidden1_biashidden2_kernelhidden2_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_8572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*B
_input_shapes1
/:���������
::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:���������

%
_user_specified_nameInput_input
�	
�
@__inference_Hidden1_layer_call_and_return_conditional_losses_791

inputs(
$matmul_readvariableop_hidden1_kernel'
#biasadd_readvariableop_hidden1_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_hidden1_kernel*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_hidden1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Elu�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
@__inference_Hidden2_layer_call_and_return_conditional_losses_813

inputs(
$matmul_readvariableop_hidden2_kernel'
#biasadd_readvariableop_hidden2_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_hidden2_kernel*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_hidden2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
A__inference_Hidden1_layer_call_and_return_conditional_losses_1078

inputs(
$matmul_readvariableop_hidden1_kernel'
#biasadd_readvariableop_hidden1_bias
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_hidden1_kernel*
_output_shapes
:	�@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
MatMul�
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_hidden1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������@2
Elu�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
� 
�
>__inference_Input_layer_call_and_return_conditional_losses_754

inputs)
%tensordot_readvariableop_input_kernel%
!biasadd_readvariableop_input_bias
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp%tensordot_readvariableop_input_kernel*
_output_shapes
:	
�*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������
2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_input_bias*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2	
BiasAddZ
EluEluBiasAdd:output:0*
T0*,
_output_shapes
:����������2
Elu�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs
�

$__inference_Input_layer_call_fn_1056

inputs
input_kernel

input_bias
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsinput_kernel
input_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *G
fBR@
>__inference_Input_layer_call_and_return_conditional_losses_7542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������

 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
Input_input8
serving_default_Input_input:0���������
;
Hidden20
StatefulPartitionedCall:0���������tensorflow/serving/predict:Չ
�#
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
*:&call_and_return_all_conditional_losses
;_default_save_signature
<__call__"�!
_tf_keras_sequential�!{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_input"}}, {"class_name": "Dense", "config": {"name": "Input", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 10]}, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "Flatten1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden2", "trainable": true, "dtype": "float32", "units": 22, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 10]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Input_input"}}, {"class_name": "Dense", "config": {"name": "Input", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 10]}, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "Flatten1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden2", "trainable": true, "dtype": "float32", "units": 22, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": [], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.01, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Input", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 10]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "Input", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 10]}, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 10]}}
�
	variables
trainable_variables
regularization_losses
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "Flatten1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Flatten1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*A&call_and_return_all_conditional_losses
B__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Hidden1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Hidden1", "trainable": true, "dtype": "float32", "units": 64, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
*C&call_and_return_all_conditional_losses
D__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "Hidden2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Hidden2", "trainable": true, "dtype": "float32", "units": 22, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
!layer_regularization_losses
"metrics
	variables
#layer_metrics

$layers
%non_trainable_variables
trainable_variables
regularization_losses
<__call__
;_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
Eserving_default"
signature_map
:	
�2Input/kernel
:�2
Input/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
&layer_regularization_losses
'metrics
	variables
(layer_metrics

)layers
*non_trainable_variables
trainable_variables
regularization_losses
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
+layer_regularization_losses
,metrics
	variables
-layer_metrics

.layers
/non_trainable_variables
trainable_variables
regularization_losses
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
!:	�@2Hidden1/kernel
:@2Hidden1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0layer_regularization_losses
1metrics
	variables
2layer_metrics

3layers
4non_trainable_variables
trainable_variables
regularization_losses
B__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
 :@2Hidden2/kernel
:2Hidden2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
5layer_regularization_losses
6metrics
	variables
7layer_metrics

8layers
9non_trainable_variables
trainable_variables
regularization_losses
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
C__inference_sequential_layer_call_and_return_conditional_losses_840
C__inference_sequential_layer_call_and_return_conditional_losses_950
C__inference_sequential_layer_call_and_return_conditional_losses_996
C__inference_sequential_layer_call_and_return_conditional_losses_826�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__wrapped_model_719�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
Input_input���������

�2�
)__inference_sequential_layer_call_fn_1007
)__inference_sequential_layer_call_fn_1018
(__inference_sequential_layer_call_fn_891
(__inference_sequential_layer_call_fn_866�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_Input_layer_call_and_return_conditional_losses_1049�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_Input_layer_call_fn_1056�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_Flatten1_layer_call_and_return_conditional_losses_1062�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_Flatten1_layer_call_fn_1067�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_Hidden1_layer_call_and_return_conditional_losses_1078�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_Hidden1_layer_call_fn_1085�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_Hidden2_layer_call_and_return_conditional_losses_1095�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_Hidden2_layer_call_fn_1102�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
!__inference_signature_wrapper_904Input_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
B__inference_Flatten1_layer_call_and_return_conditional_losses_1062^4�1
*�'
%�"
inputs����������
� "&�#
�
0����������
� |
'__inference_Flatten1_layer_call_fn_1067Q4�1
*�'
%�"
inputs����������
� "������������
A__inference_Hidden1_layer_call_and_return_conditional_losses_1078]0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� z
&__inference_Hidden1_layer_call_fn_1085P0�-
&�#
!�
inputs����������
� "����������@�
A__inference_Hidden2_layer_call_and_return_conditional_losses_1095\/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� y
&__inference_Hidden2_layer_call_fn_1102O/�,
%�"
 �
inputs���������@
� "�����������
?__inference_Input_layer_call_and_return_conditional_losses_1049e3�0
)�&
$�!
inputs���������

� "*�'
 �
0����������
� �
$__inference_Input_layer_call_fn_1056X3�0
)�&
$�!
inputs���������

� "������������
__inference__wrapped_model_719u8�5
.�+
)�&
Input_input���������

� "1�.
,
Hidden2!�
Hidden2����������
C__inference_sequential_layer_call_and_return_conditional_losses_826q@�=
6�3
)�&
Input_input���������

p

 
� "%�"
�
0���������
� �
C__inference_sequential_layer_call_and_return_conditional_losses_840q@�=
6�3
)�&
Input_input���������

p 

 
� "%�"
�
0���������
� �
C__inference_sequential_layer_call_and_return_conditional_losses_950l;�8
1�.
$�!
inputs���������

p

 
� "%�"
�
0���������
� �
C__inference_sequential_layer_call_and_return_conditional_losses_996l;�8
1�.
$�!
inputs���������

p 

 
� "%�"
�
0���������
� �
)__inference_sequential_layer_call_fn_1007_;�8
1�.
$�!
inputs���������

p

 
� "�����������
)__inference_sequential_layer_call_fn_1018_;�8
1�.
$�!
inputs���������

p 

 
� "�����������
(__inference_sequential_layer_call_fn_866d@�=
6�3
)�&
Input_input���������

p

 
� "�����������
(__inference_sequential_layer_call_fn_891d@�=
6�3
)�&
Input_input���������

p 

 
� "�����������
!__inference_signature_wrapper_904�G�D
� 
=�:
8
Input_input)�&
Input_input���������
"1�.
,
Hidden2!�
Hidden2���������