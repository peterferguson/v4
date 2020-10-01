module.exports = {
  siteTitle: 'Peter Ferguson | Data Scientist',
  siteDescription:
    'Peter Ferguson is a data scientist based in Belfast, UK who trained as a mathematical physicist but is now learning the ways of machine & deep learning.',
  siteKeywords:
    'Peter Ferguson, Peter, Ferguson, peterferguson95, data scientist, machine learning engineer, lancaster, python, cambridge, physics, maths, mathematics, mathematics for data science, mathematics for machine learning, deep learning, mathematics for deep learning',
  siteUrl: 'https://peterferguson.co.uk',
  siteLanguage: 'en_GB',
  googleAnalyticsID: process.env.GOOGLE_ANALYTICS_TRACKING_ID || 'none',
  name: 'Peter Ferguson',
  location: 'Belfast, UK',
  email: 'peterferguson95@gmail.com',
  github: 'https://github.com/bchiang7',
  twitterHandle: '',
  socialMedia: [
    {
      name: 'GitHub',
      url: 'https://github.com/peterferguson',
    },
    {
      name: 'Linkedin',
      url: 'https://www.linkedin.com/in/peter-ferguson-2041189b/',
    },
    {
      name: 'Quora',
      url: 'https://www.quora.com/profile/Peter-Ferguson-9',
    },
  ],

  navLinks: [
    {
      name: 'About',
      url: '/#about',
    },
    {
      name: 'Experience',
      url: '/#jobs',
    },
    {
      name: 'Work',
      url: '/#projects',
    },
    {
      name: 'Notes',
      url: '/notes',
    },
    {
      name: 'Contact',
      url: '/#contact',
    },
  ],

  navHeight: 100,

  colors: {
    green: '#64ffda',
    navy: '#0a192f',
    darkNavy: '#020c1b',
    atomGreen: '#52e3c2',
    atomRed: '#ff4495',
    atomBlue: '#0781ff',
    atomYellow: '#ffd900',
    atomGold: '#efb068',
    atomNavy: '#282833',
  },

  srConfig: (delay = 200) => ({
    origin: 'bottom',
    distance: '20px',
    duration: 500,
    delay,
    rotate: { x: 0, y: 0, z: 0 },
    opacity: 0,
    scale: 1,
    easing: 'cubic-bezier(0.645, 0.045, 0.355, 1)',
    mobile: true,
    reset: false,
    useDelay: 'always',
    viewFactor: 0.25,
    viewOffset: { top: 0, right: 0, bottom: 0, left: 0 },
  }),
};
