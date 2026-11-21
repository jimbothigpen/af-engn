float iq4kt_value(uint row, uint col, uint width) {
const uint nb = width/256;
const uint b = col/256;
const uint dbits = uint(data_a[row]) | (uint(data_a[row+1])<<8) |
(uint(data_a[row+2])<<16) | (uint(data_a[row+3])<<24);
float d = uintBitsToFloat(dbits);
d *= 1.0;

float s;
uint sh;
uint state;

if (b < nb) {
const uint block = row+4+128*b;
const uint ib = (col%256)/32;
const uint ig = (col%32)/4;
const uint jj = ib*8+ig;
sh = uint(data_a[block+4*ib]) | (uint(data_a[block+4*ib+1])<<8) |
(uint(data_a[block+4*ib+2])<<16) | (uint(data_a[block+4*ib+3])<<24);
const uint offset = (sh & 1u) != 0u ? 36864u : 4096u;
const int ls = int((sh & 255u)>>1) - 64;
s = d*float(ls);
const uint index = uint(data_a[block+32+jj]) |
((uint(data_a[block+96+(jj%32)]) << (8-4*(jj/32))) & 0xf00u) |
(((sh >> (8+3*ig)) & 7u) << 12);
state = index + offset;
} else {
const uint nt = (width%256)/32;
const uint tail = row+4+128*nb;
const uint it = col-nb*256;
const uint ib = it/32;
const uint ig = (it%32)/4;
const uint rec = tail+16*ib;
sh = uint(data_a[rec]) | (uint(data_a[rec+1])<<8) |
(uint(data_a[rec+2])<<16) | (uint(data_a[rec+3])<<24);
const uint offset = (sh & 1u) != 0u ? 36864u : 4096u;
const int ls = int((sh & 255u)>>1) - 64;
s = d*float(ls);
const uint index = uint(data_a[rec+4+ig]) |
(((uint(data_a[rec+12+ig/2]) >> (4*(ig&1))) & 0xfu) << 8) |
(((sh >> (8+3*ig)) & 7u) << 12);
state = index + offset;
}

for (uint k = 0; k <= col%4; ++k) {
state *= 0xCBAC1FEDu;
}
const uint bits = state & 0x3f3f3f3fu;
const int q = int((bits&255)+((bits>>8)&255)+((bits>>16)&255)+(bits>>24))-126;
return s*float(q);
}
