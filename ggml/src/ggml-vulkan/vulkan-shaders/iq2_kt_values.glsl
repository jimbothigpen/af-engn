const int iq2kt_scales[16] = int[16](-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);

float iq2kt_value(uint row, uint col, uint width) {
const uint nb = width/256;
const uint b = col/256;
const uint dbits = uint(data_a[row]) | (uint(data_a[row+1])<<8) |
(uint(data_a[row+2])<<16) | (uint(data_a[row+3])<<24);
const float d = uintBitsToFloat(dbits);

float s;
uint state;

if (b < nb) {
const uint block = row+4+68*b;
const uint in_block = col%256;
if (in_block < 128) {
const uint ib = in_block/32;
const uint ig = (in_block%32)/8;
const uint sh = uint(data_a[block+ib]);
s = d*float(iq2kt_scales[sh & 15])*1.05;
const uint qoff = block+4+2*(4*ib+ig);
const uint index = uint(data_a[qoff]) | (uint(data_a[qoff+1])<<8);
state = index+4096;
} else {
const uint ib = (in_block-128)/32;
const uint ig = ((in_block-128)%32)/8;
const uint sh = uint(data_a[block+ib]);
s = d*float(iq2kt_scales[sh >> 4])*1.05;
const uint qoff = block+4+2*(16+4*ib+ig);
const uint index = uint(data_a[qoff]) | (uint(data_a[qoff+1])<<8);
state = index+4096;
}
} else {
const uint scales = row+4+68*nb;
const uint it = col-nb*256;
const uint ib = it/32;
const uint ig = (it%32)/8;
const uint sh = uint(data_a[scales+(ib%4)]);
s = d*float(iq2kt_scales[(sh >> (4*(ib/4))) & 15])*1.05;
const uint qoff = scales+4+2*(4*ib+ig);
const uint index = uint(data_a[qoff]) | (uint(data_a[qoff+1])<<8);
state = index+4096;
}

for (uint k = 0; k <= col%8; ++k) {
state *= 0xCBAC1FEDu;
}
const uint bits = state & 0x3f3f3f3fu;
const int q = int((bits&255)+((bits>>8)&255)+((bits>>16)&255)+(bits>>24))-126;
return s*float(q);
}
