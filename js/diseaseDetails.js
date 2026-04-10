// diseaseDetails.js — bundled disease details for static hosting

const DISEASE_DETAILS = {
  Anthracnose: {
    name: 'Anthracnose',
    overview:
      'A common fungal disease that often appears during warm, humid conditions and can cause dark lesions on leaves, flowers, and fruit.',
    symptoms:
      'Look for irregular brown-to-black spots, scorched-looking patches near the margins, and tissue that dries out as lesions expand.',
    control:
      'Remove heavily affected plant material, improve airflow around the canopy, avoid prolonged leaf wetness, and follow local extension advice for fungicide timing.'
  },
  'Bacterial Canker': {
    name: 'Bacterial Canker',
    overview:
      'A bacterial infection that can affect leaves, stems, and fruit, sometimes progressing into cracking lesions and branch decline.',
    symptoms:
      'Typical signs include water-soaked or dark angular spots, raised lesions, cracking tissue, and nearby yellowing or drying.',
    control:
      'Prune and discard badly affected tissue, disinfect tools between cuts, reduce splash spread, and use locally recommended bactericide programs when needed.'
  },
  'Cutting Weevil': {
    name: 'Cutting Weevil',
    overview:
      'This class refers to insect feeding damage rather than a fungal or bacterial infection. The leaf edges can appear neatly cut or notched.',
    symptoms:
      'Leaves may show clean semicircular cuts, edge chewing, or fresh feeding marks that make the blade look trimmed.',
    control:
      'Inspect new flush regularly, remove badly damaged leaves if practical, and follow integrated pest management guidance for weevils in your growing region.'
  },
  'Die Back': {
    name: 'Die Back',
    overview:
      'Die back describes progressive drying of shoots and leaf tissue, often linked to infection, pruning wounds, or plant stress.',
    symptoms:
      'Watch for yellowing leaves, browning from the tips inward, drying twigs, and sections of growth that stop developing normally.',
    control:
      'Prune dead twigs back to healthy wood, keep trees vigorous with balanced care, avoid unnecessary wounding, and consult local disease management guidance.'
  },
  'Gall Midge': {
    name: 'Gall Midge',
    overview:
      'Gall midge damage is caused by tiny larvae that trigger raised growths in tender mango tissue, especially on young leaves.',
    symptoms:
      'Small blister-like or pimple-shaped raised spots can appear on leaves, often with distortion, curling, or reduced healthy growth.',
    control:
      'Monitor young flush closely, remove heavily infested growth where feasible, improve orchard sanitation, and time pest control according to local advice.'
  },
  Healthy: {
    name: 'Healthy',
    overview:
      'The model did not detect strong visual patterns associated with the disease classes it was trained on.',
    symptoms:
      'Healthy leaves are usually evenly green with intact margins, normal texture, and no obvious lesions, powdery coating, or insect damage.',
    control:
      'Keep up routine orchard care: balanced nutrition, irrigation management, sanitation, and regular scouting for early signs of stress.'
  },
  'Powdery Mildew': {
    name: 'Powdery Mildew',
    overview:
      'A fungal disease that commonly affects tender leaves, shoots, and blossoms, especially when humidity is high and air movement is poor.',
    symptoms:
      'You may see white or gray powdery growth on the leaf surface, distortion of young tissue, curling, or patches that later dry out.',
    control:
      'Improve airflow, avoid dense humid growth, remove badly affected tissue where possible, and follow local recommendations for mildew management.'
  },
  'Sooty Mould': {
    name: 'Sooty Mould',
    overview:
      'Sooty mould is a dark fungal coating that grows on sticky honeydew left by sap-sucking insects such as aphids, scales, or whiteflies.',
    symptoms:
      'Leaves develop a black soot-like film on the surface that can spread widely and reduce photosynthesis if insect activity continues.',
    control:
      'Focus on controlling the honeydew-producing insects first, then clean up the source of reinfestation and wash residue from leaves when practical.'
  }
};

const FALLBACK_DETAILS = {
  name: 'Details unavailable',
  overview:
    'This static GitHub Pages build includes bundled disease notes instead of a live database connection.',
  symptoms: 'No matching disease note was found for this class.',
  control: 'Review the uploaded image and compare it with reliable local agricultural guidance.'
};

export async function fetchDiseaseDetails(label) {
  return DISEASE_DETAILS[label] || { ...FALLBACK_DETAILS, name: label };
}
