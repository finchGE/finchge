import mkdocs.plugins
import mkdocs.config.config_options

class VersionPlugin(mkdocs.plugins.BasePlugin):
    def on_config(self, config, **kwargs):
        """Inject version into config."""
        try:
            import finchge
            version = finchge.__version__
        except ImportError:
            # Fallback for RTD or when package not installed
            import os
            version = os.environ.get('READTHEDOCS_VERSION', 'latest')
        
        # Add to config
        config['extra']['version'] = version
        config['extra']['is_latest'] = (version == 'latest')
        config['extra']['is_stable'] = (version == 'stable')
        
        return config