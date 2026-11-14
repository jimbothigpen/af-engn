const int iq1kt_scales[16] = int[16](-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);

float iq1kt_value(uint row, uint col, uint width) {
    const uint b = col/256;
    const uint nt = b < width/256 ? 8 : (width%256)/32;
    const uint ib = (col%256)/32;
    const uint ig = (col%32)/8;
    const uint j = 4*ib+ig;
    const uint block = row+4+56*b;
    const uint sh = uint(data_a[block+ib]);
    const uint index = uint(data_a[block+8+j]) |
        (((uint(data_a[block+8+4*nt+j%16]) >> (4*(j/16))) & 15) << 8) |
        (((sh >> (4+ig)) & 1) << 12);
    uint state = index+4096;
    for (uint k = 0; k <= col%8; ++k) {
        state *= 0xCBAC1FEDu;
    }
    const uint bits = state & 0x3f3f3f3fu;
    const int q = int((bits&255)+((bits>>8)&255)+((bits>>16)&255)+(bits>>24))-126;
    const uint dbits = uint(data_a[row]) | (uint(data_a[row+1])<<8) |
        (uint(data_a[row+2])<<16) | (uint(data_a[row+3])<<24);
    return (uintBitsToFloat(dbits)*float(iq1kt_scales[sh&15]))*float(q);
}
