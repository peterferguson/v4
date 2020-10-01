import React from 'react';
import PropTypes from 'prop-types';
import {
  IconAppStore,
  IconExternal,
  IconFolder,
  IconFork,
  IconGitHub,
  IconInstagram,
  IconInternal,
  IconLinkedin,
  IconLoader,
  IconLocation,
  IconLogo,
  IconLogoLink,
  IconPlayStore,
  IconQuora,
  IconStar,
  IconTwitter,
  IconZap,
} from '@components/icons';

const FormattedIcon = ({ name }) => {
  switch (name) {
    case 'AppStore':
      return <IconAppStore />;
    case 'External':
      return <IconExternal />;
    case 'Internal':
      return <IconInternal />;
    case 'Folder':
      return <IconFolder />;
    case 'Fork':
      return <IconFork />;
    case 'GitHub':
      return <IconGitHub />;
    case 'Instagram':
      return <IconInstagram />;
    case 'Linkedin':
      return <IconLinkedin />;
    case 'Loader':
      return <IconLoader />;
    case 'Location':
      return <IconLocation />;
    case 'Logo':
      return <IconLogo />;
    case 'LogoLink':
      return <IconLogoLink />;
    case 'PlayStore':
      return <IconPlayStore />;
    case 'Quora':
      return <IconQuora />;
    case 'Star':
      return <IconStar />;
    case 'Twitter':
      return <IconTwitter />;
    case 'Zap':
      return <IconZap />;
    default:
      return <IconExternal />;
  }
};

FormattedIcon.propTypes = {
  name: PropTypes.string.isRequired,
};

export default FormattedIcon;
