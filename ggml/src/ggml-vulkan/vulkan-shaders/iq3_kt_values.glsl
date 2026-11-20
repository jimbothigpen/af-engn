float iq3kt_value(uint row, uint col, uint width) {
const uint nb = width/256;
const uint b = col/256;
const uint dbits = uint(data_a[row]) | (uint(data_a[row+1])<<8) |
(uint(data_a[row+2])<<16) | (uint(data_a[row+3])<<24);
float d = uintBitsToFloat(dbits);
d *= 1.01;

float s;
uint state;
uint sign_byte;
uint sign_bit;

if (b < nb) {
const uint block = row+4+100*b;
const uint in_block = col%256;
if (in_block < 128) {
const uint ib = in_block/32;
const uint ig = (in_block%32)/8;
s = d*float(uint(data_a[block+ib]) & 15u);
const uint qoff = block+4+2*(4*ib+ig);
const uint index = uint(data_a[qoff]) | (uint(data_a[qoff+1])<<8);
state = index+4096;
sign_byte = uint(data_a[block+68+ig*8+(col%8)]);
sign_bit = ib;
} else {
const uint ib = (in_block-128)/32;
const uint ig = ((in_block-128)%32)/8;
s = d*float(uint(data_a[block+ib]) >> 4);
const uint qoff = block+4+2*(16+4*ib+ig);
const uint index = uint(data_a[qoff]) | (uint(data_a[qoff+1])<<8);
state = index+4096;
sign_byte = uint(data_a[block+68+ig*8+(col%8)]);
sign_bit = ib+4;
}
} else {
const uint nt = (width%256)/32;
const uint tail = row+4+100*nb;
const uint it = col-nb*256;
const uint ib = it/32;
const uint ig = (it%32)/8;
const uint qoff = tail+2*(4*ib+ig);
const uint index = uint(data_a[qoff]) | (uint(data_a[qoff+1])<<8);
state = index+4096;
const uint sh = uint(data_a[tail+12*nt+ib/2]);
s = d*float((sh >> (4*(ib&1))) & 15u);
sign_byte = uint(data_a[tail+8*nt+4*ib+ig]);
sign_bit = col%8;
}

for (uint k = 0; k <= col%8; ++k) {
state *= 0xCBAC1FEDu;
}
const uint bits = state & 0x3f3f3f3fu;
const int q = int((bits&255)+((bits>>8)&255)+((bits>>16)&255)+(bits>>24))-126;
const float mag = float(abs(q));
const float v = ((sign_byte & (1u<<sign_bit)) != 0) ? -mag : mag;
return s*v;
}
