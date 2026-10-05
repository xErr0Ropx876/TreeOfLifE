// Curated Karnataka Rice Varieties Registry
// In accordance with product honesty rules, seed genomic values are variety-level public priors,
// marked with is_placeholder = true and source_database = "local_curated".

export interface RiceVarietyProfile {
  id: string;
  canonicalName: string;
  displayName: string;
  aliases: string[];
  durationDays: number;
  grainType: string;
  releaseYear: number;
  susceptibilityVector: {
    rice_blast: number; // 0.0 (resistant) to 1.0 (highly susceptible)
    rice_brown_spot: number;
    bacterial_leaf_blight: number;
    rice_leaf_scald: number;
    rice_tungro: number;
  };
  description: string;
  commonInDistricts: string[];
}

export const KARNATAKA_RICE_VARIETIES: RiceVarietyProfile[] = [
  {
    id: 'bpt_5204',
    canonicalName: 'BPT-5204',
    displayName: 'BPT-5204 (Sona Masuri / Samba Mahsuri)',
    aliases: [
      'sona masuri',
      'sonamasuri',
      'bpt 5204',
      'bpt5204',
      'samba mahsuri',
      'samba masuri',
      'bpt'
    ],
    durationDays: 145,
    grainType: 'Medium slender (high market value)',
    releaseYear: 1982,
    susceptibilityVector: {
      rice_blast: 0.85, // Highly susceptible to blast
      rice_brown_spot: 0.70,
      bacterial_leaf_blight: 0.90, // Very susceptible to BLB
      rice_leaf_scald: 0.60,
      rice_tungro: 0.45,
    },
    description: 'Extensively cultivated across Mandya, Raichur, and Bellary; premier table variety known for aroma, but vulnerable to blast and bacterial blight.',
    commonInDistricts: ['Mandya', 'Raichur', 'Bellary', 'Davanagere', 'Mysore', 'Koppal'],
  },
  {
    id: 'ir_64',
    canonicalName: 'IR64',
    displayName: 'IR64',
    aliases: ['ir64', 'ir 64', 'ir-64'],
    durationDays: 120,
    grainType: 'Long slender',
    releaseYear: 1985,
    susceptibilityVector: {
      rice_blast: 0.35, // Moderate resistance to blast
      rice_brown_spot: 0.50,
      bacterial_leaf_blight: 0.75, // Moderate to high BLB susceptibility
      rice_leaf_scald: 0.40,
      rice_tungro: 0.60,
    },
    description: 'Widely adapted semi-dwarf variety, moderate blast tolerance, popular in irrigated river command zones.',
    commonInDistricts: ['Mandya', 'Mysore', 'Hassan', 'Davanagere'],
  },
  {
    id: 'swarna_mtu_7029',
    canonicalName: 'Swarna',
    displayName: 'Swarna (MTU-7029)',
    aliases: ['swarna', 'mtu 7029', 'mtu7029', 'mtu-7029', 'mansuri swarna'],
    durationDays: 150,
    grainType: 'Short bold',
    releaseYear: 1982,
    susceptibilityVector: {
      rice_blast: 0.55,
      rice_brown_spot: 0.75, // Susceptible to brown spot in waterlogged soils
      bacterial_leaf_blight: 0.85, // Highly susceptible to BLB
      rice_leaf_scald: 0.50,
      rice_tungro: 0.30,
    },
    description: 'High-yielding long-duration variety, sturdy tillering, prone to bacterial blight under waterlogged conditions.',
    commonInDistricts: ['Shimoga', 'Raichur', 'Bellary'],
  },
  {
    id: 'mtu_1010',
    canonicalName: 'MTU-1010',
    displayName: 'MTU-1010 (Cottondora Sannalu)',
    aliases: ['mtu 1010', 'mtu1010', 'mtu-1010', 'cottondora sannalu'],
    durationDays: 115,
    grainType: 'Long slender',
    releaseYear: 2000,
    susceptibilityVector: {
      rice_blast: 0.80, // Susceptible to blast in cool nights
      rice_brown_spot: 0.40,
      bacterial_leaf_blight: 0.65,
      rice_leaf_scald: 0.45,
      rice_tungro: 0.50,
    },
    description: 'Early maturing, non-lodging variety preferred in both kharif and rabi seasons in Karnataka canal belts.',
    commonInDistricts: ['Raichur', 'Bellary', 'Koppal', 'Mandya'],
  },
  {
    id: 'rnr_15048',
    canonicalName: 'RNR-15048',
    displayName: 'RNR-15048 (Telangana Sona)',
    aliases: ['rnr 15048', 'rnr15048', 'telangana sona', 'rnr'],
    durationDays: 125,
    grainType: 'Short slender (Low GI)',
    releaseYear: 2015,
    susceptibilityVector: {
      rice_blast: 0.25, // Good blast resistance
      rice_brown_spot: 0.55,
      bacterial_leaf_blight: 0.60,
      rice_leaf_scald: 0.35,
      rice_tungro: 0.40,
    },
    description: 'Low glycemic index variety gaining popularity in Karnataka; bred with good blast tolerance.',
    commonInDistricts: ['Bellary', 'Raichur', 'Davanagere'],
  },
  {
    id: 'pusa_basmati_1121',
    canonicalName: 'Pusa Basmati 1121',
    displayName: 'Pusa Basmati 1121',
    aliases: ['pusa 1121', 'pb 1121', 'pusa basmati', 'basmati 1121', '1121'],
    durationDays: 140,
    grainType: 'Extra long slender aromatic',
    releaseYear: 2003,
    susceptibilityVector: {
      rice_blast: 0.90, // Highly susceptible to blast
      rice_brown_spot: 0.45,
      bacterial_leaf_blight: 0.85,
      rice_leaf_scald: 0.50,
      rice_tungro: 0.55,
    },
    description: 'Premium aromatic export variety with exceptional cooked length, but highly delicate and susceptible to blast.',
    commonInDistricts: ['Shimoga', 'Davanagere'],
  },
  {
    id: 'jaya',
    canonicalName: 'Jaya',
    displayName: 'Jaya',
    aliases: ['jaya', 'iet 723'],
    durationDays: 130,
    grainType: 'Long bold',
    releaseYear: 1968,
    susceptibilityVector: {
      rice_blast: 0.50,
      rice_brown_spot: 0.60,
      bacterial_leaf_blight: 0.70,
      rice_leaf_scald: 0.40,
      rice_tungro: 0.65,
    },
    description: 'Historic benchmark high-yielding variety, resilient to varying moisture levels.',
    commonInDistricts: ['Shimoga', 'Tumkur', 'Udupi'],
  }
];

export const GENOMIC_DISCLAIMER_TEXT =
  'This is a variety-level genomic susceptibility estimate and is not DNA sequencing of this plant.';
