#ifndef RULES_HPP
#define RULES_HPP

#define RIGHT 1
#define RIGHT_DOWN 2
#define LEFT_DOWN 4
#define LEFT 8
#define LEFT_UP 16
#define RIGHT_UP 32
#define STATIONARY 64
#define BARRIER 128
#define FULL RIGHT|RIGHT_DOWN|LEFT_DOWN|LEFT|LEFT_UP|RIGHT_UP

#define RI RIGHT
#define RD RIGHT_DOWN
#define LD LEFT_DOWN
#define LE LEFT
#define LU LEFT_UP
#define RU RIGHT_UP
#define S STATIONARY
const unsigned int nRules = 2 << 8;
int rules[nRules];

void SetRules() {
	for (int i=0; i<nRules; i++)
		rules[i] = i;

	// three particle zero momentum rules
	rules[LU|LD|RI] = RU|LE|RD;
	rules[RU|LE|RD] = LU|LD|RI;
	// three particle ruless with unperturbed particle
	rules[RU|LU|LD] = LU|LE|RI;
	rules[LU|LE|RI] = RU|LU|LD;
	rules[RU|LU|RD] = RU|LE|RI;
	rules[RU|LE|RI] = RU|LU|RD;
	rules[RU|LD|RD] = LE|RD|RI;
	rules[LE|RD|RI] = RU|LD|RD;
	rules[LU|LD|RD] = LE|LD|RI;
	rules[LE|LD|RI] = LU|LD|RD;
	rules[RU|LD|RI] = LU|RD|RI;
	rules[LU|RD|RI] = RU|LD|RI;
	rules[LU|LE|RD] = RU|LE|LD;
	rules[RU|LE|LD] = LU|LE|RD;
	// two particle cyclic ruless
	rules[LE|RI] = RU|LD;
	rules[RU|LD] = LU|RD;
	rules[LU|RD] = LE|RI;
	// four particle cyclic ruless
	rules[RU|LU|LD|RD] = RU|LE|LD|RI;
	rules[RU|LE|LD|RI] = LU|LE|RD|RI;
	rules[LU|LE|RD|RI] = RU|LU|LD|RD;
	// stationary particle creation ruless
	rules[LU|RI] = RU|S;
	rules[RU|LE] = LU|S;
	rules[LU|LD] = LE|S;
	rules[LE|RD] = LD|S;
	rules[LD|RI] = RD|S;
	rules[RD|RU] = RI|S;
	rules[LU|LE|LD|RD|RI] = RU|LE|LD|RD|S;
	rules[RU|LE|LD|RD|RI] = LU|LD|RD|RI|S;
	rules[RU|LU|LD|RD|RI] = RU|LE|RD|RI|S;
	rules[RU|LU|LE|RD|RI] = RU|LU|LD|RI|S;
	rules[RU|LU|LE|LD|RI] = RU|LU|LE|RD|S;
	rules[RU|LU|LE|LD|RD] = LU|LE|LD|RI|S;
	// add all rules indexed with a stationary particle (dual rules)
	for(int i = 0;i<S;i++) {
		rules[i^(RU|LU|LE|LD|RD|RI|S)] = rules[i]^(RU|LU|LE|LD|RD|RI|S); // ^ is the exclusive or operator
	}
	// add ruless to bounce back at barriers
	for(int i = BARRIER;i<nRules;i++) {
		int highBits = i&(LE|LU|RU); // & is bitwise and operator
		int lowBits = i&(RI|RD|LD);
		rules[i] = BARRIER|(highBits>>3)|(lowBits<<3);
	}

}
#endif
