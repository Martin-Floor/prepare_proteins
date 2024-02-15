import xml.etree.ElementTree as ElementTree

class residueSelectors:

    class index:

        def __init__(self, name, residues_list):

            self.name = name
            if isinstance(residues_list, str):
                self.residues = residues_list
            elif isinstance(residues_list, list):
                self.residues = rangeToString(rangeExtract(residues_list))

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('Index')
            self.root.set('name', self.name)
            self.root.set('resnums', self.residues)

    class chainSelector:

        def __init__(self, name, chains):

            self.name = name
            self.chains = chains

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('Chain')
            self.root.set('name', self.name)
            self.root.set('chains', self.chains)

    class neighborhood:

        def __init__(self, name, selector, distance=6, include_focus_in_subset=True):

            self.name = name
            self.selector = selector
            self.distance = distance

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('Neighborhood')
            self.root.set('name', self.name)
            self.root.set('selector', self.selector)
            self.root.set('distance', str(self.distance))

    class closeContacts:

        def __init__(self, name, residue_selector, contact_threshold=4.5):

            self.name = name
            self.residue_selector = residue_selector
            self.contact_threshold = contact_threshold

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('CloseContact')
            self.root.set('name', self.name)
            self.root.set('residue_selector', self.residue_selector)
            self.root.set('contact_threshold', str(self.contact_threshold))


    class notSelector:

        def __init__(self, name, selector):

            self.name = name
            self.selector = selector

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('Not')
            self.root.set('name', self.name)
            self.root.set('selector', self.selector)

    class orSelector:

        def __init__(self, name, selectors):

            self.name = name
            self.selectors = ','.join(selectors)

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('Or')
            self.root.set('name', self.name)
            self.root.set('selectors', self.selectors)

    class andSelector:

        def __init__(self, name, selectors):

            self.name = name
            self.selectors = ','.join(selectors)

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('And')
            self.root.set('name', self.name)
            self.root.set('selectors', self.selectors)

class jumpSelectors:

    class jumpIndex:

        def __init__(self, name, jump):

            if not isinstance(jump, int):
                raise ValueError('Incorrect jump, it must be an integer.')

            self.name = name
            self.jump = jump

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('JumpIndex')
            self.root.set('name', self.name)
            self.root.set('jump', str(self.jump))

    class interchain:

        def __init__(self, name, jump):

            self.name = name

        def generateXml(self):

            self.xml = ElementTree
            self.root = self.xml.Element('Interchain')
            self.root.set('name', self.name)

def rangeExtract(lst):
    'Yield 2-tuple ranges or 1-tuple single elements from list of increasing ints'
    lenlst = len(lst)
    i = 0
    while i< lenlst:
        low = lst[i]
        while i <lenlst-1 and lst[i]+1 == lst[i+1]: i +=1
        hi = lst[i]
        if   hi - low >= 2:
            yield (low, hi)
        elif hi - low == 1:
            yield (low,)
            yield (hi,)
        else:
            yield (low,)
        i += 1

def rangeToString(ranges):
    return ( ','.join( (('%i-%i' % r) if len(r) == 2 else '%i' % r) for r in ranges ) )
