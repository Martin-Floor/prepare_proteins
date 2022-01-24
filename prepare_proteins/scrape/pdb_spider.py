import scrapy

class pdbSpider(scrapy.Spider):
    """
    Scrapy spider to retrieve information for a list of PDB ids. The PDB
    entry page is scrapped to extract information that is stored into a dictionary.
    The spider writes this dictionary into a json file.

    Attributes
    ----------
    pdb_ids : list
        List of uniprot ids to retrieve information.
    output_file : str
        Path for the output dictionary storing the retrieved information.
    """
    allowed_domains = ['www.rcsb.org/']

    def __init__(self, pdb_ids=None, output_file=None, **kwargs):
        self.pdb_ids = pdb_ids
        self.pdb_data = {}
        self.output_file = open(output_file, 'w')
        if self.pdb_ids == None:
            raise ValueError('You must give a list with the PDB IDs to retrieve\
                  information.')

    def start_requests(self):
        for pdbid in self.pdb_ids:
            yield scrapy.Request('https://www.rcsb.org/structure/'+pdbid, self.parse, meta={'pdbid': pdbid})

    def parse(self, response):

        # Get input pdb id
        current = response.meta['pdbid']
        self.pdb_data[current] = {}

        # Save scraped url
        self.pdb_data[current]['url'] = 'https://www.rcsb.org/structure/'+current

        ### Scrape PDB data here ###

        ## Basic information
        self.parseBasicInformation(response, current)

    def parseBasicInformation(self, response, current):
        ## Basic entry information ##
        structureTitle = response.css('#structureTitle::text').extract_first()
        self.pdb_data[current]['Title'] = structureTitle

        resolution = response.css('#exp_header_0_diffraction_resolution::text').extract_first()
        if resolution != None:
            resolution = float(resolution.replace('Å',''))
        self.pdb_data[current]['Resolution'] = resolution

        # Get UniProt ID
        

    def closed(self, spider):
        json.dump(self.pdb_data, self.output_file)
        self.output_file.close()
