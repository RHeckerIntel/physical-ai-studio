import { Flex, Heading, Text } from '@geti-ui/ui';

interface StepPlaceholderProps {
    title: string;
    description: string;
}

/** Empty state for a wizard step whose form has not been built yet. */
export const StepPlaceholder = ({ title, description }: StepPlaceholderProps) => (
    <Flex direction='column' gap='size-100' alignItems='center' justifyContent='center' height='100%' width='100%'>
        <Heading level={4} margin={0}>
            {title}
        </Heading>
        <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-600)', textAlign: 'center' }}>
            {description}
        </Text>
        <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-600)' }}>Coming soon.</Text>
    </Flex>
);
